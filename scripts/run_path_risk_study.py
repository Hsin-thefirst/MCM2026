"""Run a budget-matched nested-CV selection ablation in a fresh output directory."""
import argparse
from pathlib import Path
import json
import sys
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'benchmark'), str(ROOT / 'src')]
from safebench.experiment import BASE, run_development, dump
from safebench.research import compare_selection_runs, generate_selection_report
from safebench.selection import validate_selection_config


def main():
    parser = argparse.ArgumentParser(description='配对研究效用选择与路径风险约束选择；仅使用开发情景。')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/path_risk.json')
    parser.add_argument('--out', type=Path, default=BASE / 'outputs/development_path_risk_study')
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(BASE.resolve()):
        parser.error('Study output must remain inside benchmark')
    config = json.loads(args.config.read_text())
    if validate_selection_config(config) != 'path_risk':
        parser.error('Study requires a declared path-risk configuration')
    args.out.mkdir(parents=True, exist_ok=False)
    control = dict(config, selection_rule='one_se_utility')
    data = ROOT / 'data/SafeInvest_Synthetic_Data.csv'
    dump(args.out / 'study.json', dict(status='running', holdout_used=False,
        changed_factor='selection_rule only', config=config))
    run_development(control, data, args.out / 'control')
    run_development(config, data, args.out / 'constrained')
    compare_selection_runs(args.out / 'control', args.out / 'constrained', args.out / 'comparison')
    generate_selection_report(args.out / 'comparison', config)
    dump(args.out / 'study.json', dict(status='completed', holdout_used=False,
        changed_factor='selection_rule only', config=config))
    print('研究报告:', args.out / 'comparison/report.html', flush=True)


if __name__ == '__main__':
    main()
