#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
python scripts/paper_experiments_cpu.py --config configs/paper_experiments_cpu.yaml --mode all
