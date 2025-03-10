import unittest
from unittest.mock import MagicMock
import PyPDF2
import nltk
from nltk.corpus import stopwords
from io import BytesIO
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from PyPDF2 import PdfWriter, PageObject  # Ajout des imports manquants
from testCV_plusieursVect_Calculs import (
    extract_text_from_pdf, compute_euclidean_distance, 
    compute_cosine_similarity, compute_jaccard_similarity, 
    get_vectorizer, load_text
)

# Télécharger les stopwords en français
nltk.download('stopwords')
french_stopwords = stopwords.words('french')

class TestMatchingCV(unittest.TestCase):
    def create_mock_pdf(self, text):
        """Crée un fichier PDF contenant du texte (via reportlab pour garantir l'extraction)"""
        from reportlab.pdfgen import canvas  # Import ici pour éviter de le charger si non utilisé
        output = BytesIO()
        c = canvas.Canvas(output)
        c.drawString(100, 500, text)  # Ajoute du texte au PDF
        c.save()
        output.seek(0)
        return output

    def test_extract_text_from_pdf(self):
        """Test l'extraction de texte depuis un PDF mocké"""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Texte extrait d'un PDF."
        mock_pdf.pages = [mock_page]
        
        with unittest.mock.patch('PyPDF2.PdfReader', return_value=mock_pdf):
            text = extract_text_from_pdf(BytesIO(b"%PDF-1.4"))  # Simule un fichier PDF
            self.assertEqual(text, "Texte extrait d'un PDF.")

    def test_load_text_txt_utf8(self):
        """Test le chargement d'un fichier texte en UTF-8"""
        mock_file = MagicMock()
        mock_file.type = "text/plain"
        mock_file.read.return_value = b"Ceci est un texte."
        self.assertEqual(load_text(mock_file), "Ceci est un texte.")

    def test_load_text_txt_iso8859(self):
        """Test le chargement d'un fichier texte en ISO-8859-1"""
        mock_file = MagicMock()
        mock_file.type = "text/plain"
        mock_file.read.side_effect = [
            UnicodeDecodeError("utf-8", b"", 0, 1, ""),
            b"Texte encod\xe9."
        ]
        self.assertEqual(load_text(mock_file), "Texte encod\xe9.")

    def test_compute_cosine_similarity(self):
        """Test du calcul de similarité cosinus avec TfidfVectorizer"""
        text1 = "Je suis un CV."
        text2 = "Je suis une fiche de poste."
        
        vectorizer = TfidfVectorizer(stop_words=french_stopwords)
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        expected_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        computed_similarity = compute_cosine_similarity(text1, text2, vectorizer)

        self.assertAlmostEqual(computed_similarity, expected_similarity, places=6)

    def test_compute_euclidean_distance(self):
        """Test du calcul de distance euclidienne"""
        cv_text = "data science machine learning"
        job_text = "machine learning deep learning"
        vectorizer = TfidfVectorizer()
        distance = compute_euclidean_distance(cv_text, job_text, vectorizer)
        self.assertGreaterEqual(distance, 0)  # La distance euclidienne est toujours positive

    def test_compute_jaccard_similarity(self):
        """Test du calcul de similarité de Jaccard"""
        cv_text = "data science machine learning"
        job_text = "machine learning deep learning"
        similarity = compute_jaccard_similarity(cv_text, job_text)
        self.assertTrue(0 <= similarity <= 1)  # Jaccard similarity est entre 0 et 1

    def test_get_vectorizer(self):
        """Test de la récupération du bon vectorizer"""
        self.assertIsInstance(get_vectorizer('tfidf'), TfidfVectorizer)
        self.assertIsInstance(get_vectorizer('count'), CountVectorizer)
        
        with self.assertRaises(ValueError):
            get_vectorizer('invalid')

if __name__ == '__main__':
    unittest.main()
