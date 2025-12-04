import re
import string
from typing import List, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer


class ContentCleaner:
    def __init__(self, config: dict):
        self.lowercase = config.get('lowercase', True)
        self.remove_punctuation = config.get('remove_punctuation', True)
        self.remove_stopwords = config.get('remove_stopwords', True)
        self.stemming = config.get('stemming', 'porter')
        self.min_token_length = config.get('min_token_length', 2)
        self.max_token_length = config.get('max_token_length', 50)
        
        self._download_nltk_data()
        
        self.stopwords: Set[str] = set()
        if self.remove_stopwords:
            self.stopwords = set(stopwords.words('english'))
        
        self.stemmer = None
        if self.stemming == 'porter':
            self.stemmer = PorterStemmer()
        elif self.stemming == 'snowball':
            self.stemmer = SnowballStemmer('english')
        elif self.stemming == 'lemmatize':
            self.stemmer = WordNetLemmatizer()
    
    def _download_nltk_data(self):
        try:
            nltk.data.find('tokenizers/punkt_tab/english')
        except LookupError:
            try:
                nltk.download('punkt_tab', quiet=True)
            except Exception:
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
    
    def clean_content(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'\S+@\S+', '', text)
        
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
        
        if self.remove_punctuation:
            tokens = [token for token in tokens if token not in string.punctuation]
        
        tokens = [
            token for token in tokens 
            if self.min_token_length <= len(token) <= self.max_token_length
        ]
        
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        if self.stemmer:
            if self.stemming == 'lemmatize':
                tokens = [self.stemmer.lemmatize(token) for token in tokens]
            else:
                tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def clean_text_data(self, doc_id: str, text: str) -> dict:
        tokens = self.clean_content(text)
        return {
            'doc_id': doc_id,
            'text': text,
            'tokens': tokens,
            'token_count': len(tokens)
        }
    
    def clean_query_string(self, query: str) -> List[str]:
        return self.clean_content(query)


def read_data_files(file_path: str) -> List[dict]:
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
    config = {
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'stemming': 'porter',
        'min_token_length': 2,
        'max_token_length': 50
    }
    
    cleaner = ContentCleaner(config)
    
    sample_text = "Information Retrieval is the process of obtaining information system resources!"
    tokens = cleaner.clean_content(sample_text)
    print(f"Original: {sample_text}")
    print(f"Tokens: {tokens}")
