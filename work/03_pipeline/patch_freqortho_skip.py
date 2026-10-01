from pathlib import Path
p=Path('/root/FreqOrtho-SR2/test_pisasr.py')
s=p.read_text()
old="        bname = os.path.basename(image_name)\n\n        # Get caption (you can add the text prompt here)"
new="        bname = os.path.basename(image_name)\n        outpath = os.path.join(args.output_dir, bname)\n        if os.path.exists(outpath):\n            print(f'SKIP {bname}', flush=True)\n            continue\n\n        # Get caption (you can add the text prompt here)"
if 'SKIP {bname}' not in s:
    assert old in s
    s=s.replace(old,new)
p.write_text(s)
print('patched skip')
