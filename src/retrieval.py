"""
Retrieval module for query processing and document ranking.
"""

from typing import List, Tuple, Dict, Optional
import time
from preprocessor import TextPreprocessor
from indexer import InvertedIndex, BM25Ranker, TFIDFRanker


class RetrievalSystem:
    """
    Main retrieval system that coordinates preprocessing, indexing, and ranking.
    """
    
    def __init__(self, config: dict, preprocessor: TextPreprocessor, 
                 inverted_index: InvertedIndex, documents: List[dict]):
        """
        Initialize retrieval system.
        
        Args:
            config: Configuration dictionary
            preprocessor: Text preprocessor instance
            inverted_index: Inverted index instance
            documents: List of original documents (for snippets)
        """
        self.config = config
        self.preprocessor = preprocessor
        self.inverted_index = inverted_index
        self.documents = {doc['doc_id']: doc for doc in documents}
        
        # Initialize ranker based on config
        method = config['indexing']['method']
        if method == 'bm25':
            self.ranker = BM25Ranker(
                inverted_index,
                k1=config['indexing']['bm25_k1'],
                b=config['indexing']['bm25_b']
            )
        elif method == 'tfidf':
            processed_docs = [
                {'doc_id': doc['doc_id'], 'tokens': doc.get('tokens', [])}
                for doc in documents
            ]
            self.ranker = TFIDFRanker(processed_docs)
        else:
            raise ValueError(f"Unknown indexing method: {method}")
        
        self.top_k = config['retrieval']['top_k']
        self.min_score = config['retrieval']['min_score']
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Search for documents matching the query.
        
        Args:
            query: Raw query string
            top_k: Number of results to return (overrides config)
            
        Returns:
            List of result dictionaries with doc_id, score, and snippet
        """
        start_time = time.time()
        
        # Preprocess query
        query_tokens = self.preprocessor.preprocess_query(query)
        
        if not query_tokens:
            return []
        
        # Rank documents
        k = top_k if top_k is not None else self.top_k
        ranked_docs = self.ranker.rank(query_tokens, top_k=k)
        
        # Filter by minimum score
        ranked_docs = [(doc_id, score) for doc_id, score in ranked_docs 
                       if score >= self.min_score]
        
        # Prepare results with snippets
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
        """
        Generate a snippet from document text highlighting query terms.
        
        Args:
            text: Full document text
            query_tokens: Preprocessed query tokens
            max_length: Maximum snippet length
            
        Returns:
            Text snippet
        """
        if not text:
            return ""
        
        # Simple snippet: take first max_length characters
        if len(text) <= max_length:
            return text
        
        # Try to find a sentence containing query terms
        sentences = text.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(token in sentence_lower for token in query_tokens):
                snippet = sentence.strip()
                if len(snippet) <= max_length:
                    return snippet + "..."
                return snippet[:max_length] + "..."
        
        # Fallback: first max_length characters
        return text[:max_length] + "..."
    
    def batch_search(self, queries: List[str], top_k: Optional[int] = None) -> List[List[Dict]]:
        """
        Search multiple queries.
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            
        Returns:
            List of result lists, one per query
        """
        results = []
        for query in queries:
            query_results, _ = self.search(query, top_k)
            results.append(query_results)
        return results


class QueryExpander:
    """
    Query expansion using pseudo-relevance feedback.
    """
    
    def __init__(self, retrieval_system: RetrievalSystem, 
                 num_feedback_docs: int = 3, num_expansion_terms: int = 5):
        """
        Initialize query expander.
        
        Args:
            retrieval_system: RetrievalSystem instance
            num_feedback_docs: Number of top documents to use for feedback
            num_expansion_terms: Number of terms to add to query
        """
        self.retrieval_system = retrieval_system
        self.num_feedback_docs = num_feedback_docs
        self.num_expansion_terms = num_expansion_terms
    
    def expand_query(self, query: str) -> str:
        """
        Expand query using pseudo-relevance feedback.
        
        Args:
            query: Original query
            
        Returns:
            Expanded query string
        """
        # Get initial results
        results, _ = self.retrieval_system.search(query, top_k=self.num_feedback_docs)
        
        if not results:
            return query
        
        # Extract terms from top documents
        from collections import Counter
        term_freq = Counter()
        
        for result in results:
            doc_id = result['doc_id']
            doc = self.retrieval_system.documents.get(doc_id, {})
            tokens = doc.get('tokens', [])
            term_freq.update(tokens)
        
        # Get original query tokens
        query_tokens = self.retrieval_system.preprocessor.preprocess_query(query)
        
        # Remove query terms from candidates
        for token in query_tokens:
            if token in term_freq:
                del term_freq[token]
        
        # Get top expansion terms
        expansion_terms = [term for term, _ in term_freq.most_common(self.num_expansion_terms)]
        
        # Combine original and expansion terms
        expanded_query = query + " " + " ".join(expansion_terms)
        
        return expanded_query


def format_results(results: List[Dict], query: str) -> str:
    """
    Format search results for display.
    
    Args:
        results: List of result dictionaries
        query: Original query
        
    Returns:
        Formatted string
    """
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
    # Example usage
    import yaml
    from preprocessor import TextPreprocessor, load_documents
    from indexer import InvertedIndex
    
    # Load config
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Sample documents
    docs = [
        {'doc_id': 'doc1', 'text': 'Information retrieval is the process of obtaining information'},
        {'doc_id': 'doc2', 'text': 'Text mining extracts information from text'},
        {'doc_id': 'doc3', 'text': 'Machine learning is used in information retrieval systems'}
    ]
    
    # Build system
    preprocessor = TextPreprocessor(config['preprocessing'])
    processed_docs = [preprocessor.preprocess_document(d['doc_id'], d['text']) for d in docs]
    
    index = InvertedIndex()
    index.build(processed_docs)
    
    # Create retrieval system
    retrieval_system = RetrievalSystem(config, preprocessor, index, processed_docs)
    
    # Search
    results, query_time = retrieval_system.search("information retrieval")
    print(format_results(results, "information retrieval"))
    print(f"\nQuery time: {query_time:.4f} seconds")
