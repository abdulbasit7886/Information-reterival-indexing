import pickle
import os
from collections import defaultdict, Counter
from typing import List, Dict, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import math


class ReverseIndex:
    def __init__(self):
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0
        self.doc_ids: List[str] = []
        self.term_doc_freq: Dict[str, int] = {}
    
    def create_structure(self, documents: List[dict]):
        self.num_docs = len(documents)
        self.doc_ids = [doc['doc_id'] for doc in documents]
        
        for doc in documents:
            doc_id = doc['doc_id']
            tokens = doc['tokens']
            
            self.doc_lengths[doc_id] = len(tokens)
            
            term_freq = Counter(tokens)
            
            for term, freq in term_freq.items():
                self.index[term].append((doc_id, freq))
        
        self.avg_doc_length = sum(self.doc_lengths.values()) / self.num_docs
        
        for term, postings in self.index.items():
            self.term_doc_freq[term] = len(postings)
    
    def fetch_positions(self, term: str) -> List[Tuple[str, int]]:
        return self.index.get(term, [])
    
    def get_doc_freq(self, term: str) -> int:
        return self.term_doc_freq.get(term, 0)
    
    def save(self, filepath: str):
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


class ProbabilisticRanker:
    def __init__(self, inverted_index: ReverseIndex, k1: float = 1.5, b: float = 0.75):
        self.index = inverted_index
        self.k1 = k1
        self.b = b
    
    def calculate_relevance(self, query_tokens: List[str], doc_id: str) -> float:
        score = 0.0
        doc_length = self.index.doc_lengths.get(doc_id, 0)
        
        if doc_length == 0:
            return 0.0
        
        for term in query_tokens:
            tf = 0
            for posting_doc_id, freq in self.index.fetch_positions(term):
                if posting_doc_id == doc_id:
                    tf = freq
                    break
            
            if tf == 0:
                continue
            
            df = self.index.get_doc_freq(term)
            idf = math.log((self.index.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.index.avg_doc_length)
            )
            
            score += idf * (numerator / denominator)
        
        return score
    
    def order_results(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        candidate_docs = set()
        for term in query_tokens:
            for doc_id, _ in self.index.fetch_positions(term):
                candidate_docs.add(doc_id)
        
        scores = []
        for doc_id in candidate_docs:
            score = self.calculate_relevance(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]


class VectorSpaceRanker:
    def __init__(self, documents: List[dict]):
        self.doc_ids = [doc['doc_id'] for doc in documents]
        corpus = [' '.join(doc['tokens']) for doc in documents]
        
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
    
    def order_results(self, query_tokens: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        query_str = ' '.join(query_tokens)
        query_vec = self.vectorizer.transform([query_str])
        
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:
                results.append((self.doc_ids[idx], float(similarities[idx])))
        
        return results
    
    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'tfidf_matrix': self.tfidf_matrix,
                'doc_ids': self.doc_ids
            }, f)
    
    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        ranker = cls.__new__(cls)
        ranker.vectorizer = data['vectorizer']
        ranker.tfidf_matrix = data['tfidf_matrix']
        ranker.doc_ids = data['doc_ids']
        
        return ranker


if __name__ == "__main__":
    from preprocessor import ContentCleaner
    
    docs = [
        {'doc_id': 'doc1', 'text': 'Information retrieval is the process of obtaining information'},
        {'doc_id': 'doc2', 'text': 'Text mining extracts information from text'},
        {'doc_id': 'doc3', 'text': 'Machine learning is used in information retrieval'}
    ]
    
    config = {'lowercase': True, 'remove_punctuation': True, 'remove_stopwords': True, 'stemming': 'porter'}
    cleaner = ContentCleaner(config)
    processed_docs = [cleaner.clean_text_data(d['doc_id'], d['text']) for d in docs]
    
    index = ReverseIndex()
    index.create_structure(processed_docs)
    
    bm25 = ProbabilisticRanker(index)
    query = cleaner.clean_query_string("information retrieval")
    results = bm25.order_results(query, top_k=3)
    print("BM25 Results:", results)
