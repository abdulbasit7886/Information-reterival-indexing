"""
Evaluation module for measuring IR system performance.
"""

import time
import json
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict


class Evaluator:
    """
    Evaluation metrics for information retrieval systems.
    """
    
    def __init__(self, config: dict):
        """
        Initialize evaluator with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.metrics = config['evaluation']['metrics']
        self.k_values = config['evaluation']['k_values']
    
    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Precision@K.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: List of relevant document IDs
            k: Cutoff rank
            
        Returns:
            Precision@K score
        """
        if k == 0 or not retrieved:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_set = set(relevant)
        
        num_relevant = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
        
        return num_relevant / k
    
    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        """
        Calculate Recall@K.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: List of relevant document IDs
            k: Cutoff rank
            
        Returns:
            Recall@K score
        """
        if not relevant:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_set = set(relevant)
        
        num_relevant = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
        
        return num_relevant / len(relevant)
    
    def average_precision(self, retrieved: List[str], relevant: List[str]) -> float:
        """
        Calculate Average Precision (AP).
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: List of relevant document IDs
            
        Returns:
            Average Precision score
        """
        if not relevant:
            return 0.0
        
        relevant_set = set(relevant)
        num_relevant = 0
        sum_precisions = 0.0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant_set:
                num_relevant += 1
                precision_at_i = num_relevant / i
                sum_precisions += precision_at_i
        
        if num_relevant == 0:
            return 0.0
        
        return sum_precisions / len(relevant)
    
    def mean_average_precision(self, all_retrieved: List[List[str]], 
                               all_relevant: List[List[str]]) -> float:
        """
        Calculate Mean Average Precision (MAP).
        
        Args:
            all_retrieved: List of retrieved document lists (one per query)
            all_relevant: List of relevant document lists (one per query)
            
        Returns:
            MAP score
        """
        if not all_retrieved:
            return 0.0
        
        ap_scores = []
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            ap = self.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    def dcg_at_k(self, retrieved: List[str], relevant: Dict[str, float], k: int) -> float:
        """
        Calculate Discounted Cumulative Gain at K.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: Dictionary mapping doc_id to relevance score
            k: Cutoff rank
            
        Returns:
            DCG@K score
        """
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            relevance = relevant.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(self, retrieved: List[str], relevant: Dict[str, float], k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        
        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: Dictionary mapping doc_id to relevance score
            k: Cutoff rank
            
        Returns:
            NDCG@K score
        """
        dcg = self.dcg_at_k(retrieved, relevant, k)
        
        # Calculate ideal DCG
        ideal_ranking = sorted(relevant.items(), key=lambda x: x[1], reverse=True)
        ideal_retrieved = [doc_id for doc_id, _ in ideal_ranking]
        idcg = self.dcg_at_k(ideal_retrieved, relevant, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_query(self, retrieved: List[str], relevant: List[str]) -> Dict[str, float]:
        """
        Evaluate a single query with all configured metrics.
        
        Args:
            retrieved: List of retrieved document IDs
            relevant: List of relevant document IDs
            
        Returns:
            Dictionary of metric scores
        """
        results = {}
        
        for k in self.k_values:
            if 'precision' in self.metrics:
                results[f'precision@{k}'] = self.precision_at_k(retrieved, relevant, k)
            
            if 'recall' in self.metrics:
                results[f'recall@{k}'] = self.recall_at_k(retrieved, relevant, k)
        
        if 'map' in self.metrics:
            results['average_precision'] = self.average_precision(retrieved, relevant)
        
        if 'ndcg' in self.metrics:
            # Convert relevant list to dict with binary relevance
            relevant_dict = {doc_id: 1.0 for doc_id in relevant}
            for k in self.k_values:
                results[f'ndcg@{k}'] = self.ndcg_at_k(retrieved, relevant_dict, k)
        
        return results
    
    def evaluate_queries(self, all_retrieved: List[List[str]], 
                        all_relevant: List[List[str]]) -> Dict[str, float]:
        """
        Evaluate multiple queries and return average metrics.
        
        Args:
            all_retrieved: List of retrieved document lists
            all_relevant: List of relevant document lists
            
        Returns:
            Dictionary of average metric scores
        """
        all_results = []
        
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            query_results = self.evaluate_query(retrieved, relevant)
            all_results.append(query_results)
        
        # Calculate averages
        avg_results = {}
        if all_results:
            for metric in all_results[0].keys():
                scores = [r[metric] for r in all_results]
                avg_results[metric] = np.mean(scores)
                avg_results[f'{metric}_std'] = np.std(scores)
        
        # Add MAP
        if 'map' in self.metrics:
            avg_results['MAP'] = self.mean_average_precision(all_retrieved, all_relevant)
        
        return avg_results


class PerformanceBenchmark:
    """
    Performance benchmarking for IR system efficiency.
    """
    
    def __init__(self):
        self.query_times = []
        self.index_size = 0
        self.num_documents = 0
    
    def measure_query_latency(self, retrieval_system, queries: List[str]) -> Dict[str, float]:
        """
        Measure query latency statistics.
        
        Args:
            retrieval_system: RetrievalSystem instance
            queries: List of query strings
            
        Returns:
            Dictionary of latency statistics
        """
        self.query_times = []
        
        for query in queries:
            start_time = time.time()
            _, _ = retrieval_system.search(query)
            query_time = time.time() - start_time
            self.query_times.append(query_time)
        
        return {
            'mean_latency': np.mean(self.query_times),
            'median_latency': np.median(self.query_times),
            'p95_latency': np.percentile(self.query_times, 95),
            'p99_latency': np.percentile(self.query_times, 99),
            'min_latency': np.min(self.query_times),
            'max_latency': np.max(self.query_times)
        }
    
    def measure_index_size(self, index_path: str) -> Dict[str, float]:
        """
        Measure index file size.
        
        Args:
            index_path: Path to index file
            
        Returns:
            Dictionary with size metrics
        """
        import os
        
        if os.path.exists(index_path):
            size_bytes = os.path.getsize(index_path)
            size_mb = size_bytes / (1024 * 1024)
            
            return {
                'index_size_bytes': size_bytes,
                'index_size_mb': size_mb
            }
        
        return {'index_size_bytes': 0, 'index_size_mb': 0}
    
    def generate_report(self, output_path: str, effectiveness: Dict, 
                       efficiency: Dict):
        """
        Generate evaluation report.
        
        Args:
            output_path: Path to save report
            effectiveness: Effectiveness metrics
            efficiency: Efficiency metrics
        """
        report = {
            'effectiveness_metrics': effectiveness,
            'efficiency_metrics': efficiency,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def create_sample_queries() -> List[Tuple[str, List[str]]]:
    """
    Create sample queries with relevance judgments.
    
    Returns:
        List of (query, relevant_doc_ids) tuples
    """
    # This is a placeholder - in practice, you would have actual relevance judgments
    sample_queries = [
        ("information retrieval", ["doc1", "doc3"]),
        ("text mining", ["doc2"]),
        ("machine learning", ["doc3"]),
    ]
    
    return sample_queries


if __name__ == "__main__":
    # Example usage
    evaluator = Evaluator({
        'evaluation': {
            'metrics': ['precision', 'recall', 'map', 'ndcg'],
            'k_values': [5, 10, 20]
        }
    })
    
    # Sample data
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc1', 'doc3', 'doc6']
    
    # Evaluate
    results = evaluator.evaluate_query(retrieved, relevant)
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")
