import unittest
from unittest.mock import MagicMock
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from io import BytesIO

# Télécharger les stopwords en français
nltk.download('stopwords')
french_stopwords = stopwords.words('french')

# Importation des fonctions à tester
from testCV_plusieursVectorizer import extract_text_from_pdf, load_text, compute_cosine_similarity

class TestMatchingCV(unittest.TestCase):
    
    def test_extract_text_from_pdf(self):
        """Test l'extraction de texte d'un PDF"""
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Texte extrait d'un PDF."
        mock_pdf.pages = [mock_page]
        PyPDF2.PdfReader = MagicMock(return_value=mock_pdf)
        
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
        mock_file.read.side_effect = [UnicodeDecodeError("utf-8", b"", 0, 1, ""), b"Texte encod\xe9."]
        self.assertEqual(load_text(mock_file), "Texte encod\xe9.")
    
    def test_compute_cosine_similarity(self):
        """Test du calcul de similarité cosinus avec TfidfVectorizer"""
        text1 = "Je suis un CV."
        text2 = "Je suis une fiche de poste."
        
        vectorizer = TfidfVectorizer(stop_words=french_stopwords)
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        expected_similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        computed_similarity = compute_cosine_similarity(text1, text2, vectorizer_type='tfidf')
        self.assertAlmostEqual(computed_similarity, expected_similarity, places=6)

if __name__ == '__main__':
    unittest.main()