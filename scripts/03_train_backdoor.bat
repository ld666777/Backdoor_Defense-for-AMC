@echo off
chcp 65001
cd /d %~dp0\..
python main.py --config configs/quick_cpu.yaml --stage train_backdoor
pause
