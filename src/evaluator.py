import time
import json
from typing import List, Dict, Tuple
import numpy as np
from collections import defaultdict


class StatsCalculator:
    def __init__(self, config: dict):
        self.metrics = config['evaluation']['metrics']
        self.k_values = config['evaluation']['k_values']
    
    def precision_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        if k == 0 or not retrieved:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_set = set(relevant)
        
        num_relevant = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
        
        return num_relevant / k
    
    def recall_at_k(self, retrieved: List[str], relevant: List[str], k: int) -> float:
        if not relevant:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_set = set(relevant)
        
        num_relevant = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
        
        return num_relevant / len(relevant)
    
    def average_precision(self, retrieved: List[str], relevant: List[str]) -> float:
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
        if not all_retrieved:
            return 0.0
        
        ap_scores = []
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            ap = self.average_precision(retrieved, relevant)
            ap_scores.append(ap)
        
        return np.mean(ap_scores)
    
    def dcg_at_k(self, retrieved: List[str], relevant: Dict[str, float], k: int) -> float:
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], 1):
            relevance = relevant.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(self, retrieved: List[str], relevant: Dict[str, float], k: int) -> float:
        dcg = self.dcg_at_k(retrieved, relevant, k)
        
        ideal_ranking = sorted(relevant.items(), key=lambda x: x[1], reverse=True)
        ideal_retrieved = [doc_id for doc_id, _ in ideal_ranking]
        idcg = self.dcg_at_k(ideal_retrieved, relevant, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def assess_single(self, retrieved: List[str], relevant: List[str]) -> Dict[str, float]:
        results = {}
        
        for k in self.k_values:
            if 'precision' in self.metrics:
                results[f'precision@{k}'] = self.precision_at_k(retrieved, relevant, k)
            
            if 'recall' in self.metrics:
                results[f'recall@{k}'] = self.recall_at_k(retrieved, relevant, k)
        
        if 'map' in self.metrics:
            results['average_precision'] = self.average_precision(retrieved, relevant)
        
        if 'ndcg' in self.metrics:
            relevant_dict = {doc_id: 1.0 for doc_id in relevant}
            for k in self.k_values:
                results[f'ndcg@{k}'] = self.ndcg_at_k(retrieved, relevant_dict, k)
        
        return results
    
    def assess_batch(self, all_retrieved: List[List[str]], 
                        all_relevant: List[List[str]]) -> Dict[str, float]:
        all_results = []
        
        for retrieved, relevant in zip(all_retrieved, all_relevant):
            query_results = self.assess_single(retrieved, relevant)
            all_results.append(query_results)
        
        avg_results = {}
        if all_results:
            for metric in all_results[0].keys():
                scores = [r[metric] for r in all_results]
                avg_results[metric] = np.mean(scores)
                avg_results[f'{metric}_std'] = np.std(scores)
        
        if 'map' in self.metrics:
            avg_results['MAP'] = self.mean_average_precision(all_retrieved, all_relevant)
        
        return avg_results


class SystemProfiler:
    def __init__(self):
        self.query_times = []
        self.index_size = 0
        self.num_documents = 0
    
    def check_speed(self, query_processor, queries: List[str]) -> Dict[str, float]:
        self.query_times = []
        
        for query in queries:
            start_time = time.time()
            _, _ = query_processor.execute_query(query)
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
    
    def check_storage(self, index_path: str) -> Dict[str, float]:
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
        report = {
            'effectiveness_metrics': effectiveness,
            'efficiency_metrics': efficiency,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report


def create_sample_queries() -> List[Tuple[str, List[str]]]:
    sample_queries = [
        ("information retrieval", ["doc1", "doc3"]),
        ("text mining", ["doc2"]),
        ("machine learning", ["doc3"]),
    ]
    
    return sample_queries


if __name__ == "__main__":
    calculator = StatsCalculator({
        'evaluation': {
            'metrics': ['precision', 'recall', 'map', 'ndcg'],
            'k_values': [5, 10, 20]
        }
    })
    
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc1', 'doc3', 'doc6']
    
    results = calculator.assess_single(retrieved, relevant)
    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")
