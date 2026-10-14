#!/usr/bin/env python3
"""All writes default to the project benchmark folder. Run from any working directory."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True  # Also avoids bytecode writes in the existing shared package.
from safebench import ROOT
from safebench.experiment import BASE, run_development, freeze, run_final

def main():
    parser=argparse.ArgumentParser(description="SafeInvest 基准实验：开发、报告、核验、冻结与一次性最终评价。")
    parser.add_argument('command',choices=['development','report','audit','freeze','final'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/benchmark.json')
    parser.add_argument('--data',type=Path,default=ROOT/'data/SafeInvest_Synthetic_Data.csv')
    parser.add_argument('--out',type=Path,default=BASE/'outputs/development_consolidated')
    parser.add_argument('--policy',default='cost_aware')
    parser.add_argument('--bundle',type=Path,default=BASE/'frozen_policy.json')
    parser.add_argument('--development-run',type=Path,default=BASE/'outputs/development_consolidated')
    parser.add_argument('--acknowledge-final',action='store_true')
    args=parser.parse_args()
    # Keep benchmark artifacts grouped under the benchmark folder.
    for path in (args.out,args.bundle):
        if not path.resolve().is_relative_to(BASE.resolve()):
            parser.error('Output/bundle paths must remain inside benchmark')
    config=json.loads(args.config.read_text())
    if args.command=='development': run_development(config,args.data,args.out)
    elif args.command=='report':
        from safebench.report import generate_report
        generate_report(args.out)
    elif args.command=='freeze': freeze(config,args.data,args.policy,args.bundle,args.development_run)
    elif args.command=='audit':
        from safebench.validation import audit_outputs
        print(json.dumps(audit_outputs(args.out),indent=2))
    else: run_final(args.bundle,args.data,args.out,args.acknowledge_final)

if __name__=='__main__': main()
