@echo off

choco install python -y
choco install ffmpeg -y
choco install libsndfile -y
python -m pip install numpy pandas tensorflow spleeter poetry


echo Installation completed.
pause
