#!/bin/bash
# Final evaluation script
# Run after development evaluation completes

set -e

echo "Step 1: Freeze development run..."
python benchmark/run.py freeze \
  --policy drawdown_lifecycle_balanced \
  --development-run benchmark/outputs/drawdown_familygrid_v2 \
  --bundle benchmark/frozen_drawdown_lifecycle.json

echo "Step 2: Execute final evaluation on holdout scenarios 121-180..."
python benchmark/run.py final \
  --bundle benchmark/frozen_drawdown_lifecycle.json \
  --out benchmark/outputs/final_evaluation \
  --acknowledge-final

echo "Final evaluation complete. Results in benchmark/outputs/final_evaluation/"
