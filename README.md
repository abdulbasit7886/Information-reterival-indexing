Information Retrieval System 

A local Information Retrieval system supporting BM25 and TF-IDF ranking with preprocessing, indexing, searching, and evaluation.

# Create virtual environment
python -m venv venv
.\\venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
