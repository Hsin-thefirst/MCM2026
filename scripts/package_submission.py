"""Build and verify a source ZIP from frozen code and completed evaluation evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'benchmark')]
from safebench.experiment import verify_bundle


def build(development, bundle_path, final, output, research_runs=(), release_decision=None):
    development, bundle_path, final, output = map(Path, (development, bundle_path, final, output))
    raw = json.loads(bundle_path.read_text())
    bundle = verify_bundle(raw)
    dev_meta = json.loads((development/'manifest.json').read_text())
    final_meta = json.loads((final/'manifest.json').read_text())
    final_bundle = json.loads((final/'frozen_bundle.json').read_text())
    lock = ROOT/'benchmark/FINAL_EVALUATION_LOCK.json'
    if dev_meta['phase'] != 'development' or final_meta['phase'] != 'final':
        raise ValueError('Completed development and final evidence are required')
    if any(m['code_sha256'] != bundle['code_sha256'] for m in (dev_meta, final_meta)):
        raise ValueError('Evidence and frozen source differ')
    if final_bundle['bundle_sha256'] != raw['bundle_sha256'] or json.loads(lock.read_text())['bundle_sha256'] != raw['bundle_sha256']:
        raise ValueError('Final results and evaluation lock do not match the bundle')
    for run in (development, final):
        if not json.loads((run/'validation.json').read_text())['all_checks_passed']:
            raise ValueError('Accounting audit must pass before packaging')
    for name in ('submission_verification.json', 'independent_replay.json', 'independent_plan_a.json'):
        if not json.loads((development/name).read_text())['all_checks_passed']:
            raise ValueError('Independent development checks must pass')
    if not json.loads((final/'saved_final_verification.json').read_text())['all_checks_passed']:
        raise ValueError('Saved final evidence must pass independent checks')
    files = {ROOT/p for p in ('README.md', 'LOG.md', 'AGENTS.md', 'Makefile', 'pyproject.toml',
                              'requirements.lock.txt', 'benchmark/run.py', 'benchmark/experiment_registry.json',
                              'data/SafeInvest_Synthetic_Data.csv', '2026_MCM_SafeInvest.pdf')}
    for directory, patterns in [('src', ('*.py',)), ('benchmark/safebench', ('*.py',)),
                                ('configs', ('*.json',)), ('tests', ('*.py',)), ('scripts', ('*.py',)),
                                ('docs', ('*.md', '*.txt'))]:
        for pattern in patterns:
            files.update((ROOT/directory).rglob(pattern))
    # Development ledgers remain in the workspace; source and compact evidence fit the size limit.
    files.update(p for p in development.iterdir() if p.is_file() and p.suffix in ('.csv', '.json'))
    files.update(p for p in final.iterdir() if p.is_file() and (p.suffix in ('.csv', '.json') or p.name == 'decisions.csv.gz'))
    # The independent verifiers compare against immutable historical numeric references.
    for run in ('development_consolidated', 'development_plan_a_calibrated_v1', 'development_plan_c_screen_v3'):
        for name in ('scenario_metrics.csv', 'manifest.json'):
            reference = ROOT/'benchmark/outputs'/run/name
            if not reference.is_file():
                raise ValueError('Missing historical verification reference: '+str(reference))
            files.add(reference)
    files.update([bundle_path.resolve(), lock])
    compact_names = {'manifest.json', 'config.json', 'verification.json', 'decision.json',
                     'selection_verification.csv', 'cost_protection.csv', 'financial_screen.csv',
                     'summary.csv', 'tuning_trials.json', 'independent_replay.json',
                     'comparison_h30.csv', 'result_summary.json', 'original_a_inner_reproduction.json',
                     'customer_verification.json', 'release_decision.json'}
    research_paths = []
    for research in map(Path, research_runs):
        research = research.resolve()
        research.relative_to(ROOT/'benchmark/outputs')
        meta = json.loads((research/'manifest.json').read_text())
        if meta['phase'] != 'development' or meta['holdout_used']:
            raise ValueError('Research evidence must be development-only')
        research_paths.append(str(research.relative_to(ROOT)))
        files.update(p for p in research.rglob('*') if p.is_file() and p.name in compact_names)
    if release_decision is not None:
        release_decision = Path(release_decision).resolve()
        release_decision.relative_to(ROOT)
        release = json.loads(release_decision.read_text())
        if release['selected_policy'] != bundle['selected_policy'] or release['holdout_reused']:
            raise ValueError('Release decision must match the frozen financial program')
        if release['status'] != 'ready':
            raise ValueError('Release decision has not completed verification')
        files.add(release_decision)
    contents = {}
    for path in sorted(files):
        resolved = path.resolve()
        if not resolved.is_relative_to(ROOT):
            raise ValueError('Package inputs must be inside the project')
        name = str(resolved.relative_to(ROOT))
        contents[name] = path.read_bytes()
    manifest = dict(selected_policy=bundle['selected_policy'], code_sha256=bundle['code_sha256'],
        bundle_sha256=raw['bundle_sha256'], final_evaluation_completed=True,
        note='源码、冻结程序和数值证据；未包含论文。最终评价锁保留，不得重跑最终集后调参。',
        files={name:hashlib.sha256(data).hexdigest() for name,data in contents.items()})
    if research_paths:
        manifest['research_evidence'] = research_paths
        manifest['research_scope'] = '开发探索的源码、配置、紧凑数值证据；完整研究缓存/账本保留在工作区，不构成新的最终评价。'
    if release_decision is not None:
        manifest['release_decision'] = str(release_decision.relative_to(ROOT))
    contents['PACKAGE_MANIFEST.json'] = (json.dumps(manifest, indent=2, ensure_ascii=False)+'\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(contents.items()):
            info = zipfile.ZipInfo(name, (2026, 10, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    if output.stat().st_size > 50_000_000:
        raise ValueError('Source ZIP exceeds the 50 MB competition limit')
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        for name, expected in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    result = dict(path=str(output.resolve()), size_bytes=output.stat().st_size,
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(), files=len(contents),
                  all_checks_passed=True, selected_policy=bundle['selected_policy'])
    with output.with_suffix('.verification.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='打包已冻结的源码、数值证据及一次性最终评价结果')
    parser.add_argument('--development-run', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--final-run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--research-run', action='append', default=[])
    parser.add_argument('--release-decision')
    args = parser.parse_args()
    build(args.development_run, args.bundle, args.final_run, args.out, args.research_run, args.release_decision)
