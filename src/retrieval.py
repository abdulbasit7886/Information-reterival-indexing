from typing import List, Tuple, Dict, Optional
import time
from preprocessor import ContentCleaner
from indexer import ReverseIndex, ProbabilisticRanker, VectorSpaceRanker


class QueryProcessor:
    def __init__(self, config: dict, cleaner: ContentCleaner, 
                 reverse_index: ReverseIndex, documents: List[dict]):
        self.config = config
        self.cleaner = cleaner
        self.reverse_index = reverse_index
        self.documents = {doc['doc_id']: doc for doc in documents}
        
        method = config['indexing']['method']
        if method == 'bm25':
            self.ranker = ProbabilisticRanker(
                reverse_index,
                k1=config['indexing']['bm25_k1'],
                b=config['indexing']['bm25_b']
            )
        elif method == 'tfidf':
            processed_docs = [
                {'doc_id': doc['doc_id'], 'tokens': doc.get('tokens', [])}
                for doc in documents
            ]
            self.ranker = VectorSpaceRanker(processed_docs)
        else:
            raise ValueError(f"Unknown indexing method: {method}")
        
        self.top_k = config['retrieval']['top_k']
        self.min_score = config['retrieval']['min_score']
    
    def execute_query(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        start_time = time.time()
        
        query_tokens = self.cleaner.clean_query_string(query)
        
        if not query_tokens:
            return []
        
        k = top_k if top_k is not None else self.top_k
        ranked_docs = self.ranker.order_results(query_tokens, top_k=k)
        
        ranked_docs = [(doc_id, score) for doc_id, score in ranked_docs 
                       if score >= self.min_score]
        
        results = []
        for doc_id, score in ranked_docs:
            doc = self.documents.get(doc_id, {})
            snippet = self._generate_snippet(doc.get('text', ''), query_tokens)
            
            results.append({
                'doc_id': doc_id,
                'score': score,
                'snippet': snippet,
                'full_text': doc.get('text', '')
            })
        
        query_time = time.time() - start_time
        
        return results, query_time
    
    def _generate_snippet(self, text: str, query_tokens: List[str], 
                         max_length: int = 200) -> str:
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(token in sentence_lower for token in query_tokens):
                snippet = sentence.strip()
                if len(snippet) <= max_length:
                    return snippet + "..."
                return snippet[:max_length] + "..."
        
        return text[:max_length] + "..."
    
    def execute_batch(self, queries: List[str], top_k: Optional[int] = None) -> List[List[Dict]]:
        results = []
        for query in queries:
            query_results, _ = self.execute_query(query, top_k)
            results.append(query_results)
        return results


class TermExpander:
    def __init__(self, query_processor: QueryProcessor, 
                 num_feedback_docs: int = 3, num_expansion_terms: int = 5):
        self.query_processor = query_processor
        self.num_feedback_docs = num_feedback_docs
        self.num_expansion_terms = num_expansion_terms
    
    def augment_query(self, query: str) -> str:
        results, _ = self.query_processor.execute_query(query, top_k=self.num_feedback_docs)
        
        if not results:
            return query
        
        from collections import Counter
        term_freq = Counter()
        
        for result in results:
            doc_id = result['doc_id']
            doc = self.query_processor.documents.get(doc_id, {})
            tokens = doc.get('tokens', [])
            term_freq.update(tokens)
        
        query_tokens = self.query_processor.cleaner.clean_query_string(query)
        
        for token in query_tokens:
            if token in term_freq:
                del term_freq[token]
        
        expansion_terms = [term for term, _ in term_freq.most_common(self.num_expansion_terms)]
        
        expanded_query = query + " " + " ".join(expansion_terms)
        
        return expanded_query


def format_results(results: List[Dict], query: str) -> str:
    output = []
    output.append(f"\nQuery: {query}")
    output.append(f"Found {len(results)} results\n")
    output.append("=" * 80)
    
    for idx, result in enumerate(results, 1):
        output.append(f"\n{idx}. Document ID: {result['doc_id']}")
        output.append(f"   Score: {result['score']:.4f}")
        output.append(f"   Snippet: {result['snippet']}")
        output.append("-" * 80)
    
    return "\n".join(output)


if __name__ == "__main__":
    import yaml
    from preprocessor import ContentCleaner, read_data_files
    from indexer import ReverseIndex
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    docs = [
        {'doc_id': 'doc1', 'text': 'Information retrieval is the process of obtaining information'},
        {'doc_id': 'doc2', 'text': 'Text mining extracts information from text'},
        {'doc_id': 'doc3', 'text': 'Machine learning is used in information retrieval systems'}
    ]
    
    cleaner = ContentCleaner(config['preprocessing'])
    processed_docs = [cleaner.clean_text_data(d['doc_id'], d['text']) for d in docs]
    
    index = ReverseIndex()
    index.create_structure(processed_docs)
    
    processor = QueryProcessor(config, cleaner, index, processed_docs)
    
    results, query_time = processor.execute_query("information retrieval")
    print(format_results(results, "information retrieval"))
    print(f"\nQuery time: {query_time:.4f} seconds")
