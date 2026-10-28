@echo off
rem Launch the SEC filings dashboard (uses the repo's virtual environment if present).
cd /d "%~dp0"
if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" -m streamlit run app.py
) else (
  python -m streamlit run app.py
)
