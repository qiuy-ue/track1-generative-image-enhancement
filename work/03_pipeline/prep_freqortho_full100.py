from pathlib import Path
from PIL import Image
src=Path('/root/rivermind-data/full100_lq'); out=Path('/root/FreqOrtho-SR2/full100_in'); out.mkdir(exist_ok=True)
for c in range(1,101):
 p=src/f'case{c}.jpg'
 if not p.exists(): p=src/f'case{c}.png'
 im=Image.open(p).convert('RGB')
 im=im.resize((im.width//4,im.height//4),Image.Resampling.LANCZOS)
 im.save(out/f'case{c}.png')
 if c%20==0: print(c,flush=True)
print('DONE',len(list(out.glob('*.png'))))
