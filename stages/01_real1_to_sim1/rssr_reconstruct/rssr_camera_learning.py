"""Named camera feedback from existing checks; never a fitted camera recipe."""

FIELDS=('orientation_consistent','fov_adequate','role_correct','key_objects_visible')
VIEWS=('head','left_wrist','right_wrist')
POLICY='named_camera_feedback_v1'


def visual_checks(verdict,stages):
    values={};rows=verdict.get('views',[]) if isinstance(verdict,dict) else []
    for stage in stages:
        for view in VIEWS:
            found=[v for v in rows if isinstance(v,dict) and (v.get('kind'),v.get('view'))==(stage,view)]
            for field in FIELDS:
                value=found[0].get(field) if len(found)==1 else None
                values[f'{stage}/{view}/{field}']=value if type(value) is bool else None
    return values


def numeric_failures(report):
    failures=[]
    for area in ('rig','initial_optical_orientation','raster_visibility'):
        block=report.get(area,{})
        for error in block.get('errors',[]):failures.append(area+'/'+str(error))
        for row in block.get('views',[]):
            if row.get('passed') is False:
                failures.append('/'.join([area,str(row.get('kind')),str(row.get('view'))]))
        for row in block.get('issues',[]):
            failures.append('/'.join([area,str(row.get('kind')),str(row.get('view')),
                                     str(row.get('frame')),str(row.get('reason'))]))
    failures.extend('numeric/'+str(e) for e in report.get('errors',[]))
    for row in report.get('views',[]):
        key='/'.join([str(row.get('kind')),str(row.get('view'))])
        if not row.get('has_initial_visible_landmark'):failures.append(key+'/missing_landmark')
        if not row.get('projection_correspondences'):failures.append(key+'/missing_projection_evidence')
        if row.get('projection_passed') is False:failures.append(key+'/projection')
    return sorted(set(failures))


def feedback(before,after,stages):
    """Keep unknown, failed and established checks distinct across trials."""
    a=visual_checks((before or {}).get('visual_verdict'),stages)
    b=visual_checks(after.get('visual_verdict'),stages)
    na=set(numeric_failures((before or {}).get('numerical',{})))
    nb=set(numeric_failures(after.get('numerical',{})))
    established=before is not None and (before.get('visual_verdict') or {}).get('images_visible') is True
    visual_gained=[k for k in a if a[k] is False and b[k] is True] if established else []
    visual_lost=[k for k in a if a[k] is True and b[k] is not True] if established else []
    return {'policy':POLICY,'numerical_before':sorted(na),'numerical_after':sorted(nb),
            'numerical_resolved':sorted(na-nb),'numerical_regressions':sorted(nb-na) if before else [],
            'visual_gained':visual_gained,'visual_regressions':visual_lost,
            'visual_failed':[k for k,v in b.items() if v is False],
            'visual_unknown':[k for k,v in b.items() if v is None],
            'before_visual_observed':bool(established),
            'nonregression':not before or (not nb-na and not visual_lost),
            'passed':after['passed'],
            'evidence_scope':'camera_visual_and_geometry' if established else 'camera_geometry',
            'numeric_improvement_is_not_visual_pass':True}


def admission(evidence):
    return {'evidence_policy':POLICY,'evidence_level':evidence['evidence_scope'],
            'hard_before':evidence['numerical_before'], 'hard_after':evidence['numerical_after'],
            'established_check_nonregression':evidence['nonregression'],
            'camera_feedback':evidence}
