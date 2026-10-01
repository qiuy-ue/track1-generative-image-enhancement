import json,cv2,numpy as np,torch,pyiqa
from pathlib import Path
from PIL import Image
base=Path('/root/rivermind-data/plus3_teacher_full/output_dir'); cand=Path('/root/FreqOrtho-SR2/full100_teacher'); lqdir=Path('/root/rivermind-data/full100_lq')
M={n:pyiqa.create_metric(n,device='cuda') for n in ('clipiqa','maniqa','musiq','niqe')}; LP=pyiqa.create_metric('lpips',device='cuda')
def load(p):
 im=np.asarray(Image.open(p).convert('RGB')); h,w=im.shape[:2]; s=1024/max(h,w); return cv2.resize(im,(int(round(w*s)),int(round(h*s))),interpolation=cv2.INTER_AREA)
def t(im): return torch.from_numpy(im.copy()).permute(2,0,1).unsqueeze(0).float().div(255).cuda()
def nr(x):
 v={n:float(M[n](x).reshape(-1)[0]) for n in M}; return v['clipiqa']+v['maniqa']+v['musiq']/100+max(0,(10-v['niqe'])/10)
def edge(x):
 g=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(np.float32); gx=cv2.Sobel(g,cv2.CV_32F,1,0,3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,3); return cv2.magnitude(gx,gy)
rows=[]
for c in range(1,101):
 b=load(base/f'case{c}.jpg'); q=load(cand/f'case{c}.jpg'); l=load(lqdir/f'case{c}.jpg'); bt=t(b); qt=t(q); lt=t(l); dnr=nr(qt)-nr(bt); dlp=float(LP(qt,lt)-LP(bt,lt)); ded=float(np.corrcoef(edge(q).ravel(),edge(l).ravel())[0,1]-np.corrcoef(edge(b).ravel(),edge(l).ravel())[0,1]); rows.append({'case':c,'d_nr4':dnr,'d_lpips':dlp,'d_edge':ded}); print('CASE',c,round(dnr,6),round(dlp,6),round(ded,6),flush=True)
Path('/root/FreqOrtho-SR2/freq_teacher_vs_plus3_full100_1024.json').write_text(json.dumps(rows,indent=2),encoding='utf-8'); print('DONE',len(rows))
