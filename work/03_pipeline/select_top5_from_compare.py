import argparse,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--compare',required=True,help='freq_teacher_vs_plus3_full100_1024.json')
p.add_argument('--top-k',type=int,default=5)
p.add_argument('--out',required=True)
a=p.parse_args()
rows=json.loads(Path(a.compare).read_text(encoding='utf-8'))
for r in rows:
    r['utility']=1.10286*float(r['d_nr4'])+0.21747*float(r['d_edge'])
rows.sort(key=lambda r:r['utility'],reverse=True)
selected=[r for r in rows[:a.top_k] if r['utility']>0]
out={'formula':'1.10286*d_nr4 + 0.21747*d_edge','top_k':a.top_k,'selected_cases':[r['case'] for r in selected],'rows':selected}
Path(a.out).write_text(json.dumps(out,indent=2),encoding='utf-8')
print(','.join(str(r['case']) for r in selected))