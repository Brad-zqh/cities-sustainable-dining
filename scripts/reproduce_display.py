"""Redraw V203 result figures using only the repository's processed display data."""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'processed_display_data/v203'
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'figures'))
STEMS = {
    '12': 'Fig12_Restaurant_Quality_SDI_Evolution_Coverage',
    '13': 'Fig13_Six_Component_FourYear_Atlas',
    '14': 'Fig14_Restaurant_Quality_Inequality_FIXED_NUTRITION',
    '15': 'Fig15_RowGap_Expanded_600dpi',
    '16': 'Fig16_Joint_Quality_Walking_Price_Opportunity',
    '17': 'Fig17_Larger_Subgroup_Legends_600dpi',
    '18': 'Fig18b_Weighting_And_Cross_Scale_Diagnostics',
    '19': 'Fig19_Planning_Strategies',
    'S9': 'FigS9_Six_Five_Income_Comparison',
}


def sha256(path: Path) -> str:
    """Return a file digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_bundle(data: Path = DATA) -> dict[str, str]:
    """Fail closed for changed data, extra files, schemas, geometry or unsafe fields."""
    import geopandas as gpd
    import pandas as pd
    import re
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest.get('metadata_sha256', {}).items():
        path = (data/name).resolve()
        if not path.is_relative_to(data.resolve()) or sha256(path) != digest:
            raise ValueError('Changed or unsafe metadata')
    expected = {}
    forbidden_field = re.compile(r'(restaurant_id|review_id|user_id|email|phone|address|site_name|representative_building|source_row|hmac|token|password|cookie|pair_id)', re.I)
    forbidden_text = re.compile(r'(?:[A-Za-z]:[\\/]|/Users/|/home/|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,})')
    for item in manifest['files']:
        path = (data/item['path']).resolve()
        if not path.is_relative_to(data.resolve()) or path.suffix not in ('.csv','.gpkg'):
            raise ValueError(f'Invalid bundle path: {item["path"]}')
        if item['path'] in expected:
            raise ValueError('Duplicate manifest entry')
        actual = sha256(path)
        if actual != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError(f'Changed display file: {item["path"]}')
        frame = gpd.read_file(path) if path.suffix == '.gpkg' else pd.read_csv(path)
        if len(frame) != item['rows'] or list(frame.columns) != item['columns']:
            raise ValueError(f'Schema/dimension drift: {item["path"]}')
        if any(forbidden_field.search(c) for c in frame.columns):
            raise ValueError(f'Unsafe field in {item["path"]}')
        for column in frame.select_dtypes(include=['object','str']).columns:
            if frame[column].dropna().astype(str).str.contains(forbidden_text).any():
                raise ValueError(f'Unsafe text in {item["path"]}:{column}')
        if path.suffix == '.gpkg':
            if frame.crs.to_epsg() != 2326 or not frame.geometry.is_valid.all() or frame.geometry.isna().any() or frame.geometry.is_empty.any():
                raise ValueError(f'Invalid display geometry: {item["path"]}')
        expected[item['path']] = actual
    observed = {p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file() and p.name not in ('manifest.json','README.md','SHA256SUMS.csv','field_dictionary.csv')}
    if observed != set(expected):
        raise ValueError(f'Unexpected or missing data: {sorted(observed ^ set(expected))}')
    sums = pd.read_csv(data/'SHA256SUMS.csv').set_index('file')
    if sums.sha256.to_dict() != expected:
        raise ValueError('Hash-list drift')
    dictionary = data/'field_dictionary.csv'
    if dictionary.is_file():
        fields = pd.read_csv(dictionary)
        pairs = {(item['path'],column) for item in manifest['files'] for column in item['columns']}
        if set(zip(fields.file,fields.field)) != pairs or fields.duplicated(['file','field']).any():
            raise ValueError('Field dictionary is incomplete or duplicated')
    return expected


def render_one(number: str, out: Path) -> None:
    """Adapt native renderers to frozen public statistics, without private fallbacks."""
    before = validate_bundle()
    out.mkdir(parents=True)
    # Ignore any inherited external script/data locations.
    os.environ['CITIES_DATA_ROOT'] = str(DATA)
    os.environ['CITIES_FIGURE_DIR'] = str(out)
    os.environ['CITIES_V134_VISUAL_SCRIPTS'] = str(ROOT/'scripts')
    os.environ['CITIES_CENSUS_ROOT'] = str(DATA)
    os.environ['MPLBACKEND'] = 'Agg'
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    consumed: set[str] = set()

    def guard_reader(original):
        """Allow only hash-reviewed display files as empirical inputs."""
        @functools.wraps(original)
        def read(path, *args, **kwargs):
            resolved = Path(path).resolve()
            if not resolved.is_relative_to(DATA):
                raise PermissionError('External empirical input is forbidden in display mode')
            relative = resolved.relative_to(DATA).as_posix()
            if relative not in before:
                raise PermissionError(f'Unmanifested input: {relative}')
            consumed.add(relative)
            return original(path, *args, **kwargs)
        return read

    original_csv, original_geo, original_ogr = pd.read_csv, gpd.read_file, pyogrio.read_dataframe
    pd.read_csv = guard_reader(original_csv)
    gpd.read_file = guard_reader(original_geo)
    pyogrio.read_dataframe = guard_reader(original_ogr)
    def input_audit(event: str, arguments: tuple) -> None:
        if event != 'open' or not isinstance(arguments[0], (str, bytes)):
            return
        path = Path(os.fsdecode(arguments[0]))
        mode = arguments[1] or ''
        if any(flag in str(mode) for flag in ('w','a','x','+')):
            return
        if path.suffix.lower() in ('.csv','.gpkg','.parquet','.xlsx','.docx'):
            if not path.resolve().is_relative_to(DATA):
                raise PermissionError('External empirical/document input forbidden')
    sys.addaudithook(input_audit)
    import v138_reproduce_visual_polish as bridge
    bridge.AREA, bridge.TEMPORAL = DATA/'area', DATA/'temporal'
    bridge.PRICE_MARKET, bridge.JOINT = DATA/'market', DATA/'joint'
    bridge.DOWNSTREAM = DATA/('planning' if number == '19' else 'subgroups')
    bridge.CONTRASTS = DATA/'contrasts'
    bridge.AGGREGATION, bridge.OLD_ATLAS_GEOMETRY = DATA/'sensitivity', DATA/'atlas'
    bridge.DISPLAY_STRUCT, bridge.DISPLAY_DETAIL = DATA/'structure', DATA/'component_social'
    # Legacy broad checks validate unrelated private bundles. Display mode has
    # its own stronger hash/schema gate; native before/after checks use its snapshot.
    bridge.assert_inputs = lambda: before.copy()
    loader = bridge.load_original

    def public_loader(name: str):
        module, path = loader(name)
        if name == 'fig06_price_market':
            def market_frames():
                frames = {y:gpd.read_file(DATA/f'market/{y}_lsbg_price_market.gpkg') for y in module.YEARS}
                for frame in frames.values():
                    frame['restaurant_n'] = frame.active_restaurant_n
                return frames
            module.load_frames = market_frames
        elif name == 'render_evidence_composites':
            module.joint.NETWORK_DATA = DATA/'network'
            def joint_data():
                points = pd.read_csv(DATA/'joint/joint_access_point_estimates.csv')
                intervals = pd.read_csv(DATA/'joint/joint_access_dcca_block_intervals.csv')
                network = pd.read_csv(DATA/'network/network_price_inequality_point_estimates.csv')
                maps = {y:gpd.read_file(DATA/f'joint/{y}_joint_access_geometry.gpkg') for y in (2016,2021,2024)}
                return points, intervals, network.loc[network.threshold_min.eq(15)].copy(), maps
            module.joint.load_data = joint_data
            module.source_hashes = lambda: before.copy()
        elif name == 'render_quality_inequality_composite':
            module.render = functools.partial(module.render, display_only=True)
        elif name == 'fig09_social_within_year':
            module.main = functools.partial(module.main, display_contrasts=DATA/'contrasts/within_year_contrasts.csv')
        elif name == 'render_scale_weight_composite':
            module.build = functools.partial(module.build, diagnostics_only=True)
        elif name == 'render_planning_composite':
            module.source.SRC = DATA/'planning'
            module.source.JOINT_SRC = DATA/'planning'
            module.hashes = lambda: before.copy()
        return module, path

    bridge.load_original = public_loader
    import v140_reader_figure_touchups as parent
    def sdi_maps(evidence):
        frames = {y:gpd.read_file(DATA/f'area/{y}_lsbg_price_market.gpkg') for y in (2011,2016,2021,2024)}
        for frame in frames.values():
            frame[evidence.sdi.base.FIELD] = frame.sdi_equal_arithmetic
        return frames
    parent._fixed_sdi_maps = sdi_maps
    old_argv = sys.argv
    try:
        if number == 'S9':
            from figS9_display import render
            render(DATA/'regression_comparison/income_models_six_and_five.csv',out)
        elif number in ('15','17'):
            import v204_cosmetic_figure_refinements as latest
            latest.HERE = latest.OUT = out
            latest.SIX_DOMAIN_REGRESSION = DATA/'market/quality_market_decomposition.csv'
            {'15':latest.render15,'17':latest.render17}[number]()
        elif number == '13':
            sys.argv = ['display','--only','13']
            bridge.main()
        elif number in ('12','16'):
            import v147_reader_figure_refinements as prior
            sys.argv = ['display','--only',number]
            prior.main()
        else:
            import v148_panel_precision as prior
            sys.argv = ['display','--only', '18b' if number == '18' else number]
            prior.main()
    finally:
        sys.argv = old_argv
        pd.read_csv, gpd.read_file, pyogrio.read_dataframe = original_csv, original_geo, original_ogr
    if validate_bundle() != before:
        raise AssertionError('Public data changed during rendering')
    paths = [out/f'{STEMS[number]}.{ext}' for ext in ('png','pdf','svg')]
    if not all(p.is_file() and p.stat().st_size for p in paths):
        raise FileNotFoundError('Expected PNG/PDF/SVG exports missing')
    receipt = dict(figure=number,scope='V203 display reproduction; V204 typography for 15/17',
                   empirical_model_rerun=False,external_empirical_reads_allowed=False,
                   consumed_input_sha256={p:before[p] for p in sorted(consumed)},
                   output_sha256={p.name:sha256(p) for p in paths})
    (out/'public_display_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')


def main() -> int:
    """Validate the bundle or dispatch each figure in an isolated process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=[*STEMS,'all'],default='all')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.validate_only:
        inputs = validate_bundle()
        print(f'PASS: {len(inputs)} hash-locked display files')
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required and must not exist')
    out = args.output_dir.resolve()
    if out.exists() or out.is_relative_to(DATA) or DATA.is_relative_to(out):
        parser.error('Use a fresh output directory outside the input bundle')
    if args.worker:
        if args.figure == 'all':
            parser.error('Worker requires one figure')
        render_one(args.figure,out)
        return 0
    validate_bundle()
    out.mkdir(parents=True)
    for number in (STEMS if args.figure == 'all' else [args.figure]):
        print(f'Redrawing V203 Fig. {number}',flush=True)
        command = [sys.executable,'-X','utf8',str(Path(__file__).resolve()),'--worker',
                   '--figure',number,'--output-dir',str(out/number)]
        with (out/f'figure_{number}.log').open('w',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
