"""
Indexing module for building inverted index and computing TF-IDF/BM25 scores.
"""

import pickle
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import math


class InvertedIndex:
    """
    Inverted index data structure for efficient document retrieval.
    
    Attributes:
        index: Dictionary mapping terms to posting lists
        doc_lengths: Dictionary of document lengths
        avg_doc_length: Average document length
        num_docs: Total number of documents
    """
    
    def __init__(self):
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0
        self.doc_ids: List[str] = []
        self.term_doc_freq: Dict[str, int] = {}  # Document frequency per term
    
    def build(self, documents: List[dict]):
        """
        Build inverted index from preprocessed documents.
        
        Args:
            documents: List of dicts with 'doc_id' and 'tokens' fields
        """
        self.num_docs = len(documents)
        self.doc_ids = [doc['doc_id'] for doc in documents]
        
        # Build index
        for doc in documents:
            doc_id = doc['doc_id']
            tokens = doc['tokens']
            
            # Store document length
            self.doc_lengths[doc_id] = len(tokens)
            
            # Count term frequencies in this document
            term_freq = Counter(tokens)
            
            # Add to inverted index
            for term, freq in term_freq.items():
                self.index[term].append((doc_id, freq))
        
        # Calculate average document length
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs
        
        # Calculate document frequency for each term
        for term, postings in self.index.items():
            self.term_doc_freq[term] = len(postings)
    
    def get_postings(self, term: str) -> List[Tuple[str, int]]:
        """
        Get posting list for a term.
        
        Args:
            term: Query term
            
        Returns:
            List of (doc_id, term_frequency) tuples
        """
        return self.index.get(term, [])
    
    def get_doc_freq(self, term: str) -> int:
        """Get document frequency for a term."""
        return self.term_doc_freq.get(term, 0)
    
    def save(self, filepath: str):
        """Save index to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'index': dict(self.index),
                'doc_lengths': self.doc_lengths,
                'avg_doc_length': self.avg_doc_length,
                'num_docs': self.num_docs,
                'doc_ids': self.doc_ids,
                'term_doc_freq': self.term_doc_freq
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load index from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        index = cls()
        index.index = defaultdict(list, data['index'])
        index.doc_lengths = data['doc_lengths']
        index.avg_doc_length = data['avg_doc_length']
        index.num_docs = data['num_docs']
        index.doc_ids = data['doc_ids']
        index.term_doc_freq = data['term_doc_freq']
        
        return index


class BM25Ranker:
    """
    BM25 ranking algorithm implementation.
    
    BM25 is a probabilistic ranking function that considers:
    - Term frequency (with saturation)
    - Inverse document frequency
    - Document length normalization
    """
    
    def __init__(self, inverted_index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 ranker.
        
        Args:
            inverted_index: Inverted index object
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.index = inverted_index
        self.k1 = k1
        self.b = b
    
    def score(self, query_tokens: List[str], doc_id: str) -> float:
        """
        Calculate BM25 score for a document given a query.
        
        Args:
            query_tokens: List of preprocessed query tokens
            doc_id: Document identifier
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        
        if doc_length == 0:
            return 0.0
        
        for term in query_tokens:
            # Get term frequency in document
            tf = 0
            for posting_doc_id, freq in self.index.get_postings(term):
                if posting_doc_id == doc_id:
                    tf = freq
                    break
            
            if tf == 0:
                continue
            
            # Calculate IDF
            df = self.index.get_doc_freq(term)
            idf = math.log((self.index.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # Calculate BM25 component
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.index.avg_doc_length)
            )
            
            score += idf * (numerator / denominator)
        
        return score
    
    def rank(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank all documents for a query.
        
        Args:
            query_tokens: List of preprocessed query tokens
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples, sorted by score descending
        """
        # Get candidate documents (union of all postings)
        candidate_docs = set()
        for term in query_tokens:
            for doc_id, _ in self.index.get_postings(term):
                candidate_docs.add(doc_id)
        
        # Score all candidates
        scores = []
        for doc_id in candidate_docs:
            score = self.score(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class TFIDFRanker:
    """
    TF-IDF ranking using cosine similarity.
    """
    
    def __init__(self, documents: List[dict]):
        """
        Initialize TF-IDF ranker.
        
        Args:
            documents: List of preprocessed documents with 'tokens' field
        """
        # Convert tokens back to strings for TfidfVectorizer
        self.doc_ids = [doc['doc_id'] for doc in documents]
        corpus = [' '.join(doc['tokens']) for doc in documents]
        
        # Build TF-IDF matrix
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
    
    def rank(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Rank documents using TF-IDF cosine similarity.
        
        Args:
            query_tokens: List of preprocessed query tokens
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples
        """
        # Convert query to TF-IDF vector
        query_str = ' '.join(query_tokens)
        query_vec = self.vectorizer.transform([query_str])
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.doc_ids[idx], float(similarities[idx])))
        
        return results
    
    def save(self, filepath: str):
        """Save TF-IDF model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'doc_ids': self.doc_ids
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load TF-IDF model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        ranker = cls.__new__(cls)
        ranker.vectorizer = data['vectorizer']
        ranker.tfidf_matrix = data['tfidf_matrix']
        ranker.doc_ids = data['doc_ids']
        
        return ranker


if __name__ == "__main__":
    # Example usage
    from preprocessor import TextPreprocessor
    
    # Sample documents
    docs = [
        {'doc_id': 'doc1', 'text': 'Information retrieval is the process of obtaining information'},
        {'doc_id': 'doc2', 'text': 'Text mining extracts information from text'},
        {'doc_id': 'doc3', 'text': 'Machine learning is used in information retrieval'}
    ]
    
    # Preprocess
    config = {'lowercase': True, 'remove_punctuation': True, 'remove_stopwords': True, 'stemming': 'porter'}
    preprocessor = TextPreprocessor(config)
    processed_docs = [preprocessor.preprocess_document(d['doc_id'], d['text']) for d in docs]
    
    # Build index
    index = InvertedIndex()
    index.build(processed_docs)
    
    # Test BM25
    bm25 = BM25Ranker(index)
    query = preprocessor.preprocess_query("information retrieval")
    results = bm25.rank(query, top_k=3)
    print("BM25 Results:", results)
