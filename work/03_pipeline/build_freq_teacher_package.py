import io,json,zipfile,hashlib,argparse
from pathlib import Path
from PIL import Image
ap=argparse.ArgumentParser(); ap.add_argument('--base',required=True); ap.add_argument('--teacher-dir',required=True); ap.add_argument('--cases',default=''); ap.add_argument('--out',required=True); ap.add_argument('--validation',required=True); a=ap.parse_args(); base=Path(a.base); td=Path(a.teacher_dir); out=Path(a.out)
cases=set(range(1,101)) if not a.cases else {int(x) for x in a.cases.split(',') if x.strip()}
with zipfile.ZipFile(base) as zin, zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zout:
    for name in sorted(zin.namelist()):
        data=None
        if name.startswith('output_dir/case') and name.endswith('.jpg'):
            c=int(name.split('case')[-1].split('.')[0])
            if c in cases:
                p=td/f'case{c}.jpg'
                if not p.exists(): raise FileNotFoundError(p)
                data=p.read_bytes()
        zout.writestr(name,data if data is not None else zin.read(name))
with zipfile.ZipFile(out) as z:
    names=[n for n in z.namelist() if n.startswith('output_dir/case') and n.endswith('.jpg')]
    sizes={}; qlayers=[]; changed=[]; other=[]
    with zipfile.ZipFile(base) as b:
        for n in names:
            d=z.read(n); bd=b.read(n); im=Image.open(io.BytesIO(d)); rgb=im.convert('RGB'); sizes[f'{rgb.size[0]}x{rgb.size[1]}']=sizes.get(f'{rgb.size[0]}x{rgb.size[1]}',0)+1
            if d!=bd:
                c=int(n.split('case')[-1].split('.')[0]); changed.append(c); q=im.quantization; qlayers.append({'case':c,'q': [list(map(int,q[0])),list(map(int,q[1]))] if q else None,'layer':im.layer})
            elif n not in [f'output_dir/case{c}.jpg' for c in cases]: other.append(n)
    val={'base':str(base),'teacher_dir':str(td),'requested_cases':sorted(cases),'changed_cases':sorted(changed),'missing_unexpected_changes':sorted(set(cases)-set(changed)),'entries':len(names),'sizes':sizes,'other_differences':other,'zip_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'bytes':out.stat().st_size,'q93_layers':qlayers}
Path(a.validation).write_text(json.dumps(val,indent=2),encoding='utf-8'); print(json.dumps(val,indent=2))
