"""Draw current manuscript figures from archived frozen inputs; no model refitting."""
from pathlib import Path
import importlib.util, subprocess, sys, argparse
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--figures',default='12345');a=ap.parse_args()
    if any(c in a.figures for c in '1234'):
        p=ROOT/'scripts/51_refine_publication_figures.py'
        spec=importlib.util.spec_from_file_location('draw_frozen',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        for d in [m.OUT,m.PAN,m.SUP]:d.mkdir(parents=True,exist_ok=True)
        for i in a.figures:
            if i in '1234':print('Drawing Figure',i,flush=True);getattr(m,'figure'+i)()
    if '5' in a.figures:
        subprocess.run([sys.executable,str(ROOT/'scripts/59_plot_figure5_visual_redesign.py')],cwd=ROOT,check=True)
if __name__=='__main__':main()
