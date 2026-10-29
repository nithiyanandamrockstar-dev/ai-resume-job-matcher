@echo off
title AI Resume Analyzer & Job Matcher
cd /d "%~dp0"
echo ===================================================
echo   AI Resume Analyzer & Job Matcher (Glassmorphism)
echo   Local Web App URL: http://localhost:8501
echo ===================================================
echo.
"C:\Users\NIthi\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py
pause
