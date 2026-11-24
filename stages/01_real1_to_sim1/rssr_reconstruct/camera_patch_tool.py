"""Atomic camera edits selected by the Agent; no per-case override channel."""
from copy import deepcopy
from pilot_camera_agent import apply_camera_edit, CAMERAS


def apply_patch(profiles, origin, observations, patch, table_frames=None, *, visibility_context=None):
    if not isinstance(patch, dict):
        raise ValueError('Camera patch must be an object')
    if set(patch)=={'edits'}:
        edits=patch['edits']
        if not isinstance(edits,list) or not 1<=len(edits)<=6:
            raise ValueError('One to six camera edits required')
    else:
        edits=[patch]
    seen=set()
    for edit in edits:
        if not isinstance(edit,dict) or set(edit)!={'kind','view','edit'}:
            raise ValueError('Camera edit requires only kind, view and edit')
        kind,view=edit['kind'],edit['view']
        if not isinstance(kind,str) or kind not in profiles or not isinstance(view,str) or view not in CAMERAS:
            raise ValueError('Unknown camera binding')
        # Head edits affect both models in the common table frame. Reject two
        # head entries whose composition would hide a larger per-step edit.
        binding=('shared','head') if view=='head' else (kind,view)
        if binding in seen:
            raise ValueError('Repeated camera in atomic edit')
        seen.add(binding)
    candidate=deepcopy(profiles); receipts=[]
    for item in edits:
        kind,view,edit=item['kind'],item['view'],item['edit']
        if edit=={'fit_projection':True}:
            from camera_projection_fit import fit_projection
            candidate,receipt=fit_projection(candidate,origin,observations,kind,view,table_frames,
                initial_context=(visibility_context or {}).get('initial_orientation'))
        elif edit=={'fit_visibility':True}:
            from camera_visibility_fit import fit_visibility
            candidate,receipt=fit_visibility(candidate,origin,observations,kind,view,visibility_context,table_frames)
        elif edit=={'align_source_orientation':True}:
            from camera_initial_orientation import align_source_orientation
            context=(visibility_context or {}).get('initial_orientation')
            if context is None:
                raise ValueError('Source orientation transfer needs both saved initial robot states')
            candidate,receipt=align_source_orientation(candidate,origin,kind,view,context,table_frames)
        else:
            candidate=apply_camera_edit(candidate,origin,kind,view,edit,table_frames)
            receipt={'kind':kind,'view':view,'applied_edit':deepcopy(edit)}
        receipts.append(receipt)
    return candidate, {'schema':'rssr.camera_patch.receipt.v1','atomic':True,
                       'edits':receipts,'physics_and_actions_modified':False,
                       'visual_verification_required':True}
