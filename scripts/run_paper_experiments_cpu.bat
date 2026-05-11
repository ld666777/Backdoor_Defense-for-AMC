@echo off
chcp 65001
cd /d %~dp0\..
python scripts\paper_experiments_cpu.py --config configs\paper_experiments_cpu.yaml --mode all
pause
