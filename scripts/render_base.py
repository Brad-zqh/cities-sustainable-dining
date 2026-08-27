"""Re-export source-data figures with an explicit presentation-only pass."""
from pathlib import Path
import os, sys, json, hashlib, importlib, subprocess
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT
OUT = REPO/'figures/current'
JOBS = {
 3:'fig03_study_area_restaurant_distribution', 4:'fig03_openrice_fehd',
 5:'fig04_four_year_sdi', 6:'fig05_component_atlas', 7:'fig06_price_market',
 8:'fig07_joint_quality_access', 9:'fig08_weighted_rainclouds',
 10:'fig09_social_within_year', 11:'figs03_maup_maps', 12:'fig11_weight_maup',
 13:'fig12_planning', 14:'fig13_component_social_decomposition',
 15:'fig17_temporal_coverage_audit', 16:'fig18_sdi_structural_inequality'}

def hashes():
    return {p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((REPO/'source_data').rglob('*')) if p.is_file()}

def render(number):
    os.environ['CITIES_FIGURE_DIR']=str(OUT)
    os.environ['MPLBACKEND']='Agg'
    sys.path.insert(0,str(REPO/'scripts'))
    import matplotlib.figure
    from presentation_finish import finish
    old_save = matplotlib.figure.Figure.savefig
    def save(fig, path, *args, **kwargs):
        finish(fig, halos=number==12)
        return old_save(fig, path, *args, **kwargs)
    matplotlib.figure.Figure.savefig=save
    module=importlib.import_module(JOBS[number])
    for name in ('OUT','OUT_DIR','FIGURE_OUT'):
        if hasattr(module,name): setattr(module,name,OUT)
    # Legacy shared exporters retain a module-level destination.
    for modname in ('v4_plot_four_year_results','figures.v4_plot_four_year_results'):
        if modname in sys.modules: sys.modules[modname].OUT_DIR=OUT
    if number==4:
        sys.argv=[JOBS[number], '--output-base', str(OUT/'Fig4_Count_Benchmark')]
        module.main()
    elif number==3:
        module.draw()
    elif number==11:
        module.plot_maup_figure(module.load_frames('LSBG'),module.load_frames('DCCA'))
    elif number==14:
        import pandas as pd
        module.build_figure(pd.read_csv(module.SOURCE_OUT/'weighted_component_income_distributions.csv'),
                           pd.read_csv(module.SOURCE_OUT/'component_domain_reference_contrasts_2024.csv'))
    else:
        module.main()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1:
        render(int(sys.argv[1])); return
    before=hashes(); runs=[]
    for number in JOBS:
        p=subprocess.run([sys.executable,'-X','utf8',__file__,str(number)],
                         cwd=REPO,capture_output=True,text=True,encoding='utf8')
        runs.append(dict(figure=number,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        print(f'Figure {number}: {p.returncode}',flush=True)
        (ROOT/'audit/v40_figure_export.json').write_text(json.dumps(dict(runs=runs,source_sha256=before),indent=2),encoding='utf8')
        if p.returncode: raise RuntimeError(p.stderr)
    assert hashes()==before,'Source data changed'
    (ROOT/'audit/v40_figure_export.json').write_text(json.dumps(dict(status='PASS',runs=runs,
       source_files_unchanged=len(before),source_sha256=before,
       outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}),indent=2),encoding='utf8')

if __name__=='__main__': main()
