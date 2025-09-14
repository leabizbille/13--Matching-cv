@echo off
start cmd /k "uvicorn APi_MatchingCV:app --reload"
start cmd /k "streamlit run testCV_plusieursVect_Calculs.py"
