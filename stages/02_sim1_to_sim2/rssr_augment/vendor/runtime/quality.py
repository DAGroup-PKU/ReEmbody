"""Cheap full-frame anomaly scans nominate evidence; no action or pixel repair."""
import cv2,numpy as np
from common import write

def small_frames(path):
    cap=cv2.VideoCapture(str(path));frames=[]
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            h,w=bgr.shape[:2]
            frames.append(cv2.resize(bgr[round(h*135/630):round(h*495/630)],(384,96)))
    finally:cap.release()
    if not frames:raise ValueError('No decoded frames')
    return np.stack(frames).astype(np.float32)

def scan(videos,out,fps=24,seams=(),event_times=()):
    arrays={name:small_frames(path) for name,path in videos.items()}
    lengths={name:len(a) for name,a in arrays.items()}
    if len(set(lengths.values()))!=1:raise ValueError('Quality comparison has unequal frame counts')
    n=next(iter(lengths.values()))
    deltas={k:np.abs(np.diff(a,axis=0)).reshape(n-1,96,3,128,3).mean((1,3,4)) for k,a in arrays.items()}
    candidate=deltas['Real2'];reference=np.maximum(deltas['Real1'],deltas['Sim2'])
    local=np.stack([np.median(candidate[max(0,i-12):min(n-1,i+13)],axis=0) for i in range(n-1)])
    suspicious=(candidate>22)&(candidate>4*np.maximum(reference,1))&(candidate>4*np.maximum(local,1))
    peaks=np.flatnonzero(suspicious.any(axis=1))+1
    ranked=sorted(peaks,key=lambda i:float(candidate[i-1].max()),reverse=True)
    # A few paired frames are more informative than dozens of isolated thumbnails.
    focus=[]
    for idx in [*map(int,seams),*ranked]:
        if 0<idx<n and all(abs(idx-j)>4 for j in focus):focus.append(int(idx))
        if len(focus)>=3:break
    indices={0,n-1}
    for i in focus:indices.update([i-1,i,min(n-1,i+1)])
    spare=[round(i*(n-1)/7) for i in range(8)]
    for i in spare:
        if len(indices)>=14:break
        indices.add(i)
    for t in event_times:
        if len(indices)>=14:break
        indices.add(int(np.clip(round(t*fps),0,n-1)))
    violations=[]
    for i in peaks:
        violations.append({'time_s':int(i)/fps,'frame':int(i),'category':'temporal_continuity',
          'views':[v for j,v in enumerate(['head','left_wrist','right_wrist']) if suspicious[i-1,j]],
          'candidate_delta':candidate[i-1].tolist(),'reference_delta':reference[i-1].tolist()})
    duplicate=[]
    a=arrays['Real2'];s=arrays['Sim2']
    for j in [1,2]:
        gap=np.abs(a[:,:,:128]-a[:,:,128*j:128*(j+1)]).mean((1,2,3))
        source_gap=np.abs(s[:,:,:128]-s[:,:,128*j:128*(j+1)]).mean((1,2,3))
        texture=a[:,:,:128].std((1,2,3))
        ids=np.flatnonzero((gap<5)&(source_gap>20)&(texture>20))
        if len(ids)>=3:duplicate.append({'view':['head','left_wrist','right_wrist'][j], 'frames':ids[:20].tolist(),'total_flagged_frames':len(ids)})
    result={'frames':n,'fps':fps,'seam_frames':list(seams),'suspicious_jumps':violations,'possible_duplicate_views':duplicate,
      'inspection_indices':[int(i) for i in sorted(indices)],'passed':not violations and not duplicate,
      'meaning':'Conservative pixel discontinuity/duplicate-view gate, not task-success or metric motion proof.'}
    write(out,result);return result

REPAIRS={
 'temporal_continuity':'整段重新编辑唯一Sim2视频；保持连续镜头与稳定曝光，不切换源视频、不插入参考图、不重置场景。',
 'camera_identity':'逐帧保持Sim2三个不同机位；禁止将头部视图复制至左/右腕，禁止采用Real1相机投影。',
 'robot_motion':'逐帧保留Sim2灰白S1的机器人轮廓、关节、手指和运动；禁止黑色Cobot源本体替代或混入。',
 'appearance_realism':'所有视角统一使用Real1普通室内实拍曝光、白色桌面及真实背景材质；消除灰色仿真背景与硬阴影，保持S1本色。',
 'scene_cross_view':'必须同时重绘HEAD、LEFT_WRIST、RIGHT_WRIST中的任务物体，禁止只在HEAD补物体而保留腕部空桌面。三机位共享同一组物体与同一时刻状态，按各自投影保留遮挡，不复制窗口。',
 'object_motion':'按观察事件恢复物体运动；静置物体相对桌面固定，仅在真实接触后移动，释放后不跟手。',
 'contact_and_release':'补回符合Sim2既有夹爪动作的可见接触、持物和释放，不能靠改动作或物体漂移造成功。',
 'temporal_order':'保留Sim2逐帧时序；只恢复该时刻Real1物体状态，不提前或跳过交互。',
 'phase_outcome':'恢复真实演示对应的物体终态；若与冻结动作不兼容，不得用改动作或剪辑伪造成功。'}

def repair_constraints(report,automated=None):
    """Free-form LLM repair instructions cannot rewrite robot/camera controls."""
    kinds={v.get('category') for v in report.get('violations',[]) if isinstance(v,dict)}
    for k,d in report.get('criteria',{}).items():
        if isinstance(d,dict) and d.get('pass') is False:kinds.add(k)
    if automated and automated.get('suspicious_jumps'):kinds.add('temporal_continuity')
    if automated and automated.get('possible_duplicate_views'):kinds.add('camera_identity')
    return '\n'.join(REPAIRS[k] for k in REPAIRS if k in kinds)
