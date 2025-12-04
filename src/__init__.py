"""Package initialization for IR system."""

__version__ = "1.0.0"
__author__ = "CS 516 Student"

from .preprocessor import TextPreprocessor, load_documents
from .indexer import InvertedIndex, BM25Ranker, TFIDFRanker
from .retrieval import RetrievalSystem, QueryExpander, format_results
from .evaluator import Evaluator, PerformanceBenchmark

__all__ = [
    'TextPreprocessor',
    'load_documents',
    'InvertedIndex',
    'BM25Ranker',
    'TFIDFRanker',
    'RetrievalSystem',
    'QueryExpander',
    'format_results',
    'Evaluator',
    'PerformanceBenchmark'
]
