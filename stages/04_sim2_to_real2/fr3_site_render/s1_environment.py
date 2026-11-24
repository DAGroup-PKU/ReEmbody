"""Adapt the user desk asset in an isolated copy; the source is never saved."""
import argparse
import hashlib
import json
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix,Vector

TOP='Desk / 1200 x 600 x 18 mm top'
DESK_COLLECTION='01 Desk - editable geometry'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bounds(obj):
    points=np.asarray([tuple(obj.matrix_world@Vector(v)) for v in obj.bound_box])
    return points.min(0),points.max(0)


def set_table_height(scene,top):
    bpy.context.view_layer.update()
    tabletop=scene.objects.get(TOP)
    if tabletop is None:raise ValueError('Missing desk tabletop: '+TOP)
    lower,upper=bounds(tabletop);delta=float(top-upper[2]);changed=[]
    for o in scene.objects:
        if not any(c.name==DESK_COLLECTION for c in o.users_collection):continue
        if o.name==TOP or o.name=='Desk / lower edge band' or 'tabletop support' in o.name or o.name.startswith('Crank /') or o.name.startswith('Desktop /'):
            o.location.z+=delta;changed.append(o.name)
        elif 'upper sliding post' in o.name:
            length=float(o.dimensions.z);assert length+delta>0
            o.location.z+=delta/2;o.scale.z*=(length+delta)/length;changed.append(o.name)
    bpy.context.view_layer.update();lower,upper=bounds(tabletop)
    assert abs(upper[2]-top)<2e-6 and np.allclose(upper[:2]-lower[:2],[1.2,.6],atol=2e-6)
    return {'top_m':float(upper[2]),'width_m':float(upper[0]-lower[0]),'depth_m':float(upper[1]-lower[1]),'changed_objects':changed}


def apply_overrides(scene,description):
    """Baseline environment fitting only; these edits remain fixed in all episodes."""
    for name,values in description.get('objects',{}).items():
        obj=scene.objects[name]
        for key,value in values.items():
            if key=='matrix_world':obj.matrix_world=Matrix(value)
            elif key in {'location','rotation_euler','scale'}:setattr(obj,key,value)
            elif key=='hide_render':obj.hide_render=bool(value)
            else:raise ValueError('Unsupported environment fitting field: '+key)
    for name,values in description.get('materials',{}).items():
        mat=bpy.data.materials[name];assert mat.use_nodes
        shader=next((n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None);assert shader
        for key,value in values.items():
            if key=='base_color':shader.inputs['Base Color'].default_value=(*value,1)
            elif key=='roughness':shader.inputs['Roughness'].default_value=float(value)
            else:raise ValueError('Unsupported baseline material field: '+key)
    if 'exposure_ev' in description:scene.view_settings.exposure=float(description['exposure_ev'])
    bpy.context.view_layer.update()


def lights(scene):
    rows=[]
    for o in scene.objects:
        if o.type!='LIGHT':continue
        direction=o.rotation_euler.to_matrix()@Vector((0,0,-1))
        rows.append({'name':o.name,'position_m':list(o.location),'target_m':list(o.location+direction*2),
            'energy_w':float(o.data.energy),'color':list(o.data.color),'size_m':[float(o.data.size),float(o.data.size_y) if o.data.shape=='RECTANGLE' else float(o.data.size)]})
    return rows


def prepare(source,output,top,overrides=None):
    source=Path(source).resolve();output=Path(output).resolve();assert source!=output and not output.exists()
    source_sha=sha(source);bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False)
    scene=bpy.data.scenes['Scene'];bpy.context.window.scene=scene
    for o in scene.objects:o.update_tag()
    scene.frame_set(scene.frame_current);bpy.context.view_layer.update()
    table=set_table_height(scene,top);apply_overrides(scene,overrides or {})
    output.parent.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output),compress=True)
    receipt={'schema':'rssr.s1_heightfit.user_desk_environment.v1','source':str(source),'source_sha256':source_sha,
        'derived_blend':str(output),'derived_sha256':sha(output),'scene':'Scene','table':table,'lights':lights(scene),
        'baseline_overrides':overrides or {},'robot_or_cups_in_user_scene':False,'source_unchanged':sha(source)==source_sha,
        'blender_version':bpy.app.version_string,'original_file_blender_version':'5.2.44','native_runtime_compatibility_reopen_pending':True,
        'policy':'User authorized baseline environment fitting; per-episode augmentation changes lights only'}
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--top',type=float,required=True);p.add_argument('--overrides',type=Path);a=p.parse_args()
    r=prepare(a.source,a.output,a.top,json.loads(a.overrides.read_text()) if a.overrides else None)
    print(json.dumps(r,ensure_ascii=False))


def apply_style_fit(scene,style):
    """Fixed shared materials/lights and their procedural graphs, never per-view edits."""
    if not style:return {}
    audit={}
    if style.get('neutralize_lights'):
        for obj in scene.objects:
            if obj.type=='LIGHT' and obj.data.type=='AREA':obj.data.color=(1,1,1)
        audit['area_light_rgb']=[1,1,1]
    if 'primary_light_transform' in style:
        description=style['primary_light_transform'];name=description.get('name','Light / broad left office softbox');obj=scene.objects[name];assert obj.type=='LIGHT' and obj.data.type=='AREA'
        obj.location=description['position_m'];obj.rotation_euler=(Vector(description['target_m'])-obj.location).to_track_quat('-Z','Y').to_euler()
        if 'power_w' in description:obj.data.energy=float(description['power_w'])
        audit['primary_light_transform']=dict(name=name,position_m=list(obj.location),target_m=description['target_m'],power_w=float(obj.data.energy))
    added=style.get('baseline_added_area_lights',{})
    for name in added:
        obj=scene.objects.get(name)
        if obj is None:
            data=bpy.data.lights.new(name,'AREA');obj=bpy.data.objects.new(name,data);scene.collection.objects.link(obj)
        assert obj.type=='LIGHT' and obj.data.type=='AREA'
    fitted_lights=dict(style.get('baseline_area_lights',{}))
    assert not set(fitted_lights)&set(added)
    fitted_lights.update(added)
    if fitted_lights:
        fitted=[]
        for name,description in fitted_lights.items():
            obj=scene.objects[name]
            assert obj.type=='LIGHT' and obj.data.type=='AREA'
            assert set(description)<={'position_m','target_m','power_w','size_m','color_rgb'}
            if 'position_m' in description:obj.location=description['position_m']
            if 'target_m' in description:obj.rotation_euler=(Vector(description['target_m'])-obj.location).to_track_quat('-Z','Y').to_euler()
            if 'power_w' in description:
                power=float(description['power_w']);assert 0<=power<=3000;obj.data.energy=power
            if 'size_m' in description:
                size=description['size_m'];assert len(size)==2 and all(0<float(v)<=6 for v in size)
                obj.data.shape='RECTANGLE';obj.data.size=float(size[0]);obj.data.size_y=float(size[1])
            if 'color_rgb' in description:
                color=description['color_rgb'];assert len(color)==3 and all(0<=float(v)<=1 for v in color)
                obj.data.color=color
            fitted.append(dict(name=name,position_m=list(obj.location),power_w=float(obj.data.energy),
                size_m=[float(obj.data.size),float(obj.data.size_y)],color_rgb=list(obj.data.color)))
        audit['baseline_area_lights']=fitted
        audit['baseline_added_area_light_names']=list(added)
    table=bpy.data.materials.get('Desktop / warm white polished laminate')
    if table and 'table_color_ramp' in style:
        ramps=[node for node in table.node_tree.nodes if node.type=='VALTORGB'];assert len(ramps)==1
        colors=style['table_color_ramp'];assert len(colors)==len(ramps[0].color_ramp.elements)
        for element,color in zip(ramps[0].color_ramp.elements,colors):element.color=(*color,1)
        audit['table_color_ramp']=colors
    gray=bpy.data.materials.get('Divider / upper warm gray panel')
    if gray and ('gray_panel_value_scale' in style or 'gray_panel_rgb_scale' in style):
        tree=gray.node_tree;shader=next(n for n in tree.nodes if n.type=='BSDF_PRINCIPLED');socket=shader.inputs['Base Color']
        name='Heightfit gray neutral value multiply';node=tree.nodes.get(name)
        if node is None:
            node=tree.nodes.new('ShaderNodeMixRGB');node.name=name;node.blend_type='MULTIPLY';node.inputs[0].default_value=1
            links=list(socket.links)
            if links:tree.links.new(links[0].from_socket,node.inputs[1])
            else:node.inputs[1].default_value=socket.default_value
            tree.links.new(node.outputs[0],socket)
        if 'gray_panel_rgb_scale' in style:
            color=style['gray_panel_rgb_scale'];assert len(color)==3 and all(0<float(v)<=5 for v in color)
            node.inputs[2].default_value=(*map(float,color),1)
        else:
            scale=float(style['gray_panel_value_scale']);node.inputs[2].default_value=(scale,scale*.197/.180,scale*.197/.163,1)
        audit['gray_neutral_multiply']=list(node.inputs[2].default_value)
    for mat in bpy.data.materials:
        if not mat.use_nodes:continue
        shader=next((n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
        if shader is None:continue
        if mat.name.startswith('shared_kraft_paper') and 'cup_outer_base_color' in style:
            assert not shader.inputs['Base Color'].is_linked
            shader.inputs['Base Color'].default_value=(*style['cup_outer_base_color'],1)
            shader.inputs['Specular IOR Level'].default_value=float(style.get('cup_outer_specular_ior',.23));audit['cup_outer_base_color']=style['cup_outer_base_color']
        if mat.name.startswith('S1_native_finger_pad') and 'native_finger_base_color' in style:
            shader.inputs['Base Color'].default_value=(*style['native_finger_base_color'],1)
            shader.inputs['Roughness'].default_value=float(style.get('native_finger_roughness',.78))
            shader.inputs['Specular IOR Level'].default_value=float(style.get('native_finger_specular_ior',.18));audit['native_finger_base_color']=style['native_finger_base_color']
    return audit
