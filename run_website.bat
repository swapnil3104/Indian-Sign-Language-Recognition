@echo off
echo ====================================================
echo  Indian Sign Language (ISL) Recognition Web App
echo ====================================================
echo Starting Web Server on http://localhost:5000 ...
timeout /t 2 /nobreak >nul
start "" http://localhost:5000
"C:\Users\Asus\anaconda3\envs\isl\python.exe" app.py
pause
