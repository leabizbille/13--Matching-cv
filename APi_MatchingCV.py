from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
from nltk.corpus import stopwords
from typing import List
import uvicorn
import secrets

# Sécurité HTTP Basic
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = "admin"
    correct_password = "monmotdepasse"
    if not (secrets.compare_digest(credentials.username, correct_username) and
            secrets.compare_digest(credentials.password, correct_password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Identifiants invalides",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Télécharger les stopwords français
nltk.download('stopwords')
french_stopwords = stopwords.words('french')

app = FastAPI(
    title="API Dispatch CV",
    docs_url="/docs",
    redoc_url="/redoc"
)  # Swagger accessible publiquement

# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # éviter les None
            text += page_text
    return text

# Fonction pour lire le texte d'un fichier
def load_text(file):
    if file.filename.endswith(".pdf"):
        return extract_text_from_pdf(file.file)
    else:
        return file.file.read().decode("utf-8")

# Fonction de calcul de similarité
def compute_cosine_similarity(cv_text, job_text, vectorizer_type='tfidf'):
    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words=french_stopwords)
    elif vectorizer_type == 'count':
        vectorizer = CountVectorizer(stop_words=french_stopwords)
    elif vectorizer_type == 'hashing':
        vectorizer = HashingVectorizer(stop_words=french_stopwords, alternate_sign=False)
    else:
        raise ValueError("Vectorizer invalide : choisir 'tfidf', 'count' ou 'hashing'.")
    tfidf_matrix = vectorizer.fit_transform([cv_text, job_text])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

# Endpoint pour calculer la similarité (protégé par HTTP Basic)
@app.post("/match/")
async def match_cvs_with_job(
    cv_files: List[UploadFile] = File(...),
    job_file: UploadFile = File(...),
    vectorizer_type: str = "tfidf",
    username: str = Depends(authenticate)  # authentication obligatoire ici
):
    job_text = load_text(job_file)
    results = []

    for cv_file in cv_files:
        cv_text = load_text(cv_file)
        similarity_score = compute_cosine_similarity(cv_text, job_text, vectorizer_type)
        results.append({
            "CV": cv_file.filename,
            "Similarité Cosinus": similarity_score
        })

    return {"results": results, "user": username}

# Exécution de l'API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
