@echo off
call .venv\Scripts\activate.bat
python -m src.prepare_data --max-samples 3000
python -m src.build_index
python -m src.train --config configs/train.yaml
python -m src.evaluate --max-test-samples 100
pause
