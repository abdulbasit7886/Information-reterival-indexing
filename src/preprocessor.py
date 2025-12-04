"""
Preprocessing module for text normalization and tokenization.
Handles case folding, punctuation removal, stopword filtering, and stemming.
"""

import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer


class TextPreprocessor:
    """
    Text preprocessing pipeline for information retrieval.
    
    Attributes:
        lowercase (bool): Convert text to lowercase
        remove_punctuation (bool): Remove punctuation marks
        remove_stopwords (bool): Filter out stopwords
        stemming (str): Stemming method ('porter', 'snowball', 'lemmatize', 'none')
        min_token_length (int): Minimum token length to keep
        max_token_length (int): Maximum token length to keep
    """
    
    def __init__(self, config: dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.lowercase = config.get('lowercase', True)
        self.remove_punctuation = config.get('remove_punctuation', True)
        self.remove_stopwords = config.get('remove_stopwords', True)
        self.stemming = config.get('stemming', 'porter')
        self.min_token_length = config.get('min_token_length', 2)
        self.max_token_length = config.get('max_token_length', 50)
        
        # Download required NLTK data
        self._download_nltk_data()
        
        # Initialize stopwords
        self.stopwords: Set[str] = set()
        if self.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        
        # Initialize stemmer/lemmatizer
        self.stemmer = None
        if self.stemming == 'porter':
            self.stemmer = PorterStemmer()
        elif self.stemming == 'snowball':
            self.stemmer = SnowballStemmer('english')
        elif self.stemming == 'lemmatize':
            self.stemmer = WordNetLemmatizer()
    
    def _download_nltk_data(self):
        """Download required NLTK datasets."""
        # Some NLTK distributions provide a 'punkt_tab' tokenizer resource
        # (used by newer PunktTokenizer internals). Try to ensure both
        # 'punkt_tab' and 'punkt' are available.
        try:
            nltk.data.find('tokenizers/punkt_tab/english')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                # best-effort download; continue to try 'punkt' as fallback
                pass

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        if self.stemming == 'lemmatize':
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
    
    def preprocess(self, text: str) -> List[str]:
        """
        Apply full preprocessing pipeline to text.
        
        Args:
            text: Raw input text
            
        Returns:
            List of preprocessed tokens
        """
        # Lowercase conversion
        if self.lowercase:
            text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Tokenization. If tokenizers are missing, attempt to download and retry.
        try:
            tokens = word_tokenize(text)
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception:
                pass
            try:
                nltk.download('punkt', quiet=True)
            except Exception:
                pass
            tokens = word_tokenize(text)
        
        # Remove punctuation
        if self.remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]
        
        # Filter by length
        tokens = [
            token for token in tokens 
            if self.min_token_length <= len(token) <= self.max_token_length
        ]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Apply stemming/lemmatization
        if self.stemmer:
            if self.stemming == 'lemmatize':
                tokens = [self.stemmer.lemmatize(token) for token in tokens]
            else:
                tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def preprocess_document(self, doc_id: str, text: str) -> dict:
        """
        Preprocess a document and return structured data.
        
        Args:
            doc_id: Document identifier
            text: Document text
            
        Returns:
            Dictionary with doc_id, original text, and tokens
        """
        tokens = self.preprocess(text)
        return {
            'doc_id': doc_id,
            'text': text,
            'tokens': tokens,
            'token_count': len(tokens)
        }
    
    def preprocess_query(self, query: str) -> List[str]:
        """
        Preprocess a query (same pipeline as documents).
        
        Args:
            query: Query string
            
        Returns:
            List of preprocessed query tokens
        """
        return self.preprocess(query)


def load_documents(file_path: str) -> List[dict]:
    """
    Load documents from a file.
    Supports TXT (one doc per line) and JSON formats.
    
    Args:
        file_path: Path to document file
        
    Returns:
        List of documents with 'doc_id' and 'text' fields
    """
    import json
    import os
    
    documents = []
    
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = data
            elif isinstance(data, dict):
                documents = [{'doc_id': k, 'text': v} for k, v in data.items()]
    
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    documents.append({
                        'doc_id': f'doc_{idx}',
                        'text': line
                    })
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return documents


if __name__ == "__main__":
    # Example usage
    config = {
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'stemming': 'porter',
        'min_token_length': 2,
        'max_token_length': 50
    }
    
    preprocessor = TextPreprocessor(config)
    
    # Test preprocessing
    sample_text = "Information Retrieval is the process of obtaining information system resources!"
    tokens = preprocessor.preprocess(sample_text)
    print(f"Original: {sample_text}")
    print(f"Tokens: {tokens}")
