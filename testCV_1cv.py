import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Fonction pour extraire du texte d'un fichier PDF
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Fonction pour charger le texte d'un fichier
def load_text(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)  # Extraction du texte pour les fichiers PDF
    else:
        try:
            return file.read().decode("utf-8")  # Tentative de lecture en UTF-8
        except UnicodeDecodeError:
            return file.read().decode("ISO-8859-1")  # Si une erreur se produit, essayer en ISO-8859-1

# Fonction pour calculer la similarité cosinus
def compute_cosine_similarity(cv_text, job_description_text):
    # On vectorise les textes
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([cv_text, job_description_text])
    
    # On calcule la similarité cosinus
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cosine_sim[0][0]

# Personnalisation du thème de Streamlit
st.set_page_config(page_title="Matching CV & Fiche de Poste", page_icon=":clipboard:", layout="wide")

# Définir des couleurs et du style
primary_color = "#0072B5"  # Bleu pour les boutons et en-têtes
background_color = "#F5F5F5"  # Gris clair pour l'arrière-plan
text_color = "#333333"  # Texte sombre pour une meilleure lisibilité
highlight_color = "#FF8C00"  # Couleur orange pour les éléments importants

# Application du fond de la page
st.markdown(f"""
    <style>
        .reportview-container {{
            background-color: {background_color};
        }}
        .sidebar .sidebar-content {{
            background-color: {primary_color};
        }}
        .stButton>button {{
            background-color: {primary_color};
            color: white;
        }}
        .stFileUploader>label {{
            background-color: {highlight_color};
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 12px;
        }}
        h1, h2, h3 {{
            color: {primary_color};
        }}
        .stTextInput>div>input {{
            border-radius: 8px;
        }}
    </style>
""", unsafe_allow_html=True)

# Titre de l'application
st.title('Matching CV - Fiche de Poste')
st.markdown("**Utilisez cet outil pour comparer un CV avec une fiche de poste à l'aide de la similarité cosinus.**")

# Créer deux colonnes pour afficher les éléments côte à côte
col1, col2 = st.columns(2)

# Dépôt du CV dans la première colonne
with col1:
    st.header('Déposez votre CV')
    cv_file = st.file_uploader("Téléchargez votre CV (PDF ou texte)", type=["pdf", "txt"], key="cv_uploader")
    if cv_file is not None:
        st.success("Fichier CV chargé avec succès! 🎉")

# Dépôt de la fiche de poste dans la deuxième colonne
with col2:
    st.header('Déposez la fiche de poste')
    job_file = st.file_uploader("Téléchargez la fiche de poste (PDF ou texte)", type=["pdf", "txt"], key="job_uploader")
    if job_file is not None:
        st.success("Fichier fiche de poste chargé avec succès! 🎉")

# Vérifier si les fichiers sont téléchargés
if cv_file is not None and job_file is not None:
    # Chargement des textes
    cv_text = load_text(cv_file)
    job_description_text = load_text(job_file)
    
    # Calcul de la similarité
    similarity_score = compute_cosine_similarity(cv_text, job_description_text)
    
    # Affichage du score de similarité
    st.subheader('Résultat de la similarité cosinus')
    st.markdown(f"**Le score de similarité entre le CV et la fiche de poste est de :**")
    st.write(f"📝 **Score** : {similarity_score:.4f}")
else:
    st.warning("Veuillez télécharger un CV et une fiche de poste pour procéder au calcul.")
