```batch
@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo Generating simulation data...
python data/generate_simulation_data.py
echo Training and testing...
python run/train.py
echo Done.
pause