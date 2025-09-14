import subprocess
# python run_all.py

uvicorn_process = subprocess.Popen(
    ["uvicorn", "APi_MatchingCV:app", "--reload", "--port", "8001"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)


# Lancer Streamlit
streamlit_process = subprocess.Popen(
    ["streamlit", "run", "testCV_plusieursVect_Calculs.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Lire et afficher les logs de l'API FastAPI
for line in iter(uvicorn_process.stdout.readline, ""):
    print("[UVICORN] " + line, end="")  # Préfixe les logs avec [UVICORN]

# Lire et afficher les erreurs de FastAPI
for line in iter(uvicorn_process.stderr.readline, ""):
    print("[UVICORN ERROR] " + line, end="")  # Préfixe les erreurs avec [UVICORN ERROR]

# Lire et afficher les logs de Streamlit
for line in iter(streamlit_process.stdout.readline, ""):
    print("[STREAMLIT] " + line, end="")  # Préfixe les logs avec [STREAMLIT]

# Lire et afficher les erreurs de Streamlit
for line in iter(streamlit_process.stderr.readline, ""):
    print("[STREAMLIT ERROR] " + line, end="")  # Préfixe les erreurs avec [STREAMLIT ERROR]

# Optionnel : attendre que les deux processus se terminent
uvicorn_process.wait()
streamlit_process.wait()
