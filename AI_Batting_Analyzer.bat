@echo off
cd /d "%~dp0"
call venv312\Scripts\activate
streamlit run app_final.py
