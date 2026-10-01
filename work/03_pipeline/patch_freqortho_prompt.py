from pathlib import Path
p=Path('/root/FreqOrtho-SR2/test_pisasr.py')
s=p.read_text()
s=s.replace("validation_prompt = ''","validation_prompt = args.prompt")
if "parser.add_argument('--prompt'" not in s:
    s=s.replace("    args = parser.parse_args()", "    parser.add_argument('--prompt', type=str, default='')\n    args = parser.parse_args()")
p.write_text(s)
print('patched')
