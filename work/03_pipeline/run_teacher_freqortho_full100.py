import argparse,json,os,sys,time
from pathlib import Path
import numpy as np,torch
from PIL import Image, ImageFilter
from torchvision import transforms
WS=Path('/root/rivermind-data'); REPO=WS/'TADSR-main'; sys.path.insert(0,str(REPO)); os.chdir(REPO)
from tadsr_tilefix import TADSR_test
from my_utils.wavelet_color_fix import adain_color_fix
PROMPTS=json.loads((WS/'full100_ram_birdplant_ref_prompts.json').read_text()); SCENE=json.loads((WS/'scene_4k.json').read_text()); T800=set(SCENE['t800']); SUF=', high quality, highly detailed, sharp focus, clean, natural colors'
class A: pass
def cfg():
 a=A(); p=WS/'TADSR'; a.pretrained_model_name_or_path=str(p); a.tadsr_path=str(WS/'lora_svd_merge_fine/tadsr_imag2006_a15.pkl'); a.mixed_precision='fp16'; a.fast_decoder=True; a.merge_and_unload_lora=False; a.vae_encoder_tiled_size=512; a.vae_decoder_tiled_size=64; a.latent_tiled_size=96; a.latent_tiled_overlap=32; return a
def prep(half):
 w,h=half.size; target=int(round(max(w,h)/4)); scale=target/max(w,h); small=half.resize((max(8,int(round(w*scale))),max(8,int(round(h*scale)))),Image.Resampling.LANCZOS); ref=small.resize((small.width*4,small.height*4),Image.Resampling.LANCZOS); ref=ref.resize((ref.width-ref.width%8,ref.height-ref.height%8)); return ref,transforms.ToTensor()(ref).unsqueeze(0)
def enhance(model,image,prompt,ts,feather):
 w,h=image.size; hor=w>=h; mid=(w if hor else h)//2; patches=[image.crop((0,0,mid+feather,h)),image.crop((mid-feather,0,w,h))] if hor else [image.crop((0,0,w,mid+feather)),image.crop((0,mid-feather,w,h))]; offs=[0,mid-feather]; rend=[]
 for p in patches:
  ref,t=prep(p)
  with torch.inference_mode(): out=model(t.cuda()*2-1,prompt=prompt,timesteps=ts)
  tile=transforms.ToPILImage()(out[0].cpu()*0.5+0.5); tile=adain_color_fix(target=tile,source=ref); rend.append(np.asarray(tile,np.float32)); torch.cuda.empty_cache()
 canvas=np.zeros((h,w,3),np.float32)
 if hor:
  canvas[:,offs[0]:offs[0]+rend[0].shape[1]]=rend[0]; canvas[:,offs[1]:]=rend[1]; lo,hi=mid-feather,mid+feather; ramp=np.linspace(0,1,hi-lo,np.float32)[None,:,None]; canvas[:,lo:hi]=(1-ramp)*rend[0][:,lo-offs[0]:]+ramp*rend[1][:,:hi-offs[1]]
 else:
  canvas[offs[0]:offs[0]+rend[0].shape[0],:]=rend[0]; canvas[offs[1]:,:]=rend[1]; lo,hi=mid-feather,mid+feather; ramp=np.linspace(0,1,hi-lo,np.float32)[:,None,None]; canvas[lo:hi,:]=(1-ramp)*rend[0][lo-offs[0]:,:]+ramp*rend[1][:hi-offs[1],:]
 return Image.fromarray(np.clip(canvas,0,255).astype(np.uint8))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--cases',type=int,nargs='+',default=list(range(1,101))); ap.add_argument('--out',type=Path,required=True); ap.add_argument('--teacher',type=Path,default=Path('/root/FreqOrtho-SR2/full100_out')); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
 model=TADSR_test(cfg()).eval(); start=time.time(); done=[]
 for c in a.cases:
  dst=a.out/f'case{c}.jpg'
  if dst.exists(): done.append(c); print('SKIP',c,flush=True); continue
  lq=WS/'full100_lq'/f'case{c}.jpg'; tp=a.teacher/f'case{c}.png'; lqim=Image.open(lq).convert('RGB'); t=Image.open(tp).convert('RGB').resize(lqim.size); cond=t.filter(ImageFilter.GaussianBlur(radius=3.0)); out=enhance(model,cond,PROMPTS[f'case{c}']+SUF,800 if c in T800 else 500,64 if c in T800 else 32); out.save(dst,'JPEG',quality=93,subsampling=0); done.append(c); print('DONE',c,round(time.time()-start,1),flush=True)
 print('FULL100_TEACHER_DONE',len(done),flush=True)
if __name__=='__main__': main()
