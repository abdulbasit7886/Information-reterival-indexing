"""
Main entry point for the Information Retrieval System.
Provides command-line interface for indexing, searching, and evaluation.
"""

import argparse
import yaml
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessor import TextPreprocessor, load_documents
from indexer import InvertedIndex, BM25Ranker, TFIDFRanker
from retrieval import RetrievalSystem, format_results
from evaluator import Evaluator, PerformanceBenchmark


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_index(args, config):
    """Build and save the inverted index."""
    print("=" * 80)
    print("BUILDING INDEX")
    print("=" * 80)
    
    # Load documents
    print(f"\nLoading documents from: {args.data_dir}")
    
    # Support multiple file formats
    documents = []
    data_path = Path(args.data_dir)
    
    if data_path.is_file():
        # Single file
        documents = load_documents(str(data_path))
    elif data_path.is_dir():
        # Directory - load all .txt and .json files
        for file_path in data_path.glob('*'):
            if file_path.suffix in ['.txt', '.json']:
                docs = load_documents(str(file_path))
                documents.extend(docs)
    else:
        print(f"Error: {args.data_dir} not found!")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    # Preprocess documents
    print("\nPreprocessing documents...")
    preprocessor = TextPreprocessor(config['preprocessing'])
    
    processed_docs = []
    for doc in documents:
        processed_doc = preprocessor.preprocess_document(doc['doc_id'], doc['text'])
        processed_docs.append(processed_doc)
    
    print(f"Preprocessed {len(processed_docs)} documents")
    
    # Build inverted index
    print("\nBuilding inverted index...")
    inverted_index = InvertedIndex()
    inverted_index.build(processed_docs)
    
    print(f"Index contains {len(inverted_index.index)} unique terms")
    print(f"Average document length: {inverted_index.avg_doc_length:.2f} tokens")
    
    # Save index
    output_dir = args.output or config['paths']['index_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, 'inverted_index.pkl')
    inverted_index.save(index_path)
    print(f"\nSaved inverted index to: {index_path}")
    
    # Save processed documents
    docs_path = os.path.join(output_dir, 'processed_docs.pkl')
    import pickle
    with open(docs_path, 'wb') as f:
        pickle.dump(processed_docs, f)
    print(f"Saved processed documents to: {docs_path}")
    
    # Build and save TF-IDF model if configured
    if config['indexing']['method'] in ['tfidf', 'hybrid']:
        print("\nBuilding TF-IDF model...")
        tfidf_ranker = TFIDFRanker(processed_docs)
        tfidf_path = os.path.join(output_dir, 'tfidf_model.pkl')
        tfidf_ranker.save(tfidf_path)
        print(f"Saved TF-IDF model to: {tfidf_path}")
    
    print("\n" + "=" * 80)
    print("INDEX BUILD COMPLETE")
    print("=" * 80)


def search_interactive(args, config):
    """Interactive search interface."""
    print("=" * 80)
    print("INTERACTIVE SEARCH")
    print("=" * 80)
    
    # Load index and documents
    index_dir = args.index or config['paths']['index_dir']
    
    print(f"\nLoading index from: {index_dir}")
    index_path = os.path.join(index_dir, 'inverted_index.pkl')
    inverted_index = InvertedIndex.load(index_path)
    
    docs_path = os.path.join(index_dir, 'processed_docs.pkl')
    import pickle
    with open(docs_path, 'rb') as f:
        processed_docs = pickle.load(f)
    
    print(f"Loaded index with {len(inverted_index.index)} terms")
    print(f"Loaded {len(processed_docs)} documents")
    
    # Initialize retrieval system
    preprocessor = TextPreprocessor(config['preprocessing'])
    retrieval_system = RetrievalSystem(config, preprocessor, inverted_index, processed_docs)
    
    print("\n" + "=" * 80)
    print("Ready to search! (Type 'quit' to exit)")
    print("=" * 80)
    
    # Interactive loop
    while True:
        query = input("\nEnter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        # Search
        results, query_time = retrieval_system.search(query, top_k=args.top_k)
        
        # Display results
        print(format_results(results, query))
        print(f"\nQuery time: {query_time*1000:.2f} ms")


def search_single(args, config):
    """Search with a single query."""
    # Load index and documents
    index_dir = args.index or config['paths']['index_dir']
    
    index_path = os.path.join(index_dir, 'inverted_index.pkl')
    inverted_index = InvertedIndex.load(index_path)
    
    docs_path = os.path.join(index_dir, 'processed_docs.pkl')
    import pickle
    with open(docs_path, 'rb') as f:
        processed_docs = pickle.load(f)
    
    # Initialize retrieval system
    preprocessor = TextPreprocessor(config['preprocessing'])
    retrieval_system = RetrievalSystem(config, preprocessor, inverted_index, processed_docs)
    
    # Search
    results, query_time = retrieval_system.search(args.query, top_k=args.top_k)
    
    # Display results
    print(format_results(results, args.query))
    print(f"\nQuery time: {query_time*1000:.2f} ms")


def evaluate_system(args, config):
    """Evaluate the IR system."""
    print("=" * 80)
    print("SYSTEM EVALUATION")
    print("=" * 80)
    
    # Load index and documents
    index_dir = args.index or config['paths']['index_dir']
    
    print(f"\nLoading index from: {index_dir}")
    index_path = os.path.join(index_dir, 'inverted_index.pkl')
    inverted_index = InvertedIndex.load(index_path)
    
    docs_path = os.path.join(index_dir, 'processed_docs.pkl')
    import pickle
    with open(docs_path, 'rb') as f:
        processed_docs = pickle.load(f)
    
    # Initialize retrieval system
    preprocessor = TextPreprocessor(config['preprocessing'])
    retrieval_system = RetrievalSystem(config, preprocessor, inverted_index, processed_docs)
    
    # Load queries
    if args.queries:
        print(f"\nLoading queries from: {args.queries}")
        with open(args.queries, 'r') as f:
            query_data = json.load(f)
    else:
        # Use sample queries
        print("\nUsing sample queries (no query file provided)")
        query_data = [
            {"query": "information retrieval", "relevant": []},
            {"query": "text mining", "relevant": []},
            {"query": "machine learning", "relevant": []}
        ]
    
    # Initialize evaluator
    evaluator = Evaluator(config)
    benchmark = PerformanceBenchmark()
    
    # Run queries and collect results
    print(f"\nRunning {len(query_data)} queries...")
    all_retrieved = []
    all_relevant = []
    queries = []
    
    for item in query_data:
        query = item['query']
        relevant = item.get('relevant', [])
        
        queries.append(query)
        results, _ = retrieval_system.search(query)
        retrieved = [r['doc_id'] for r in results]
        
        all_retrieved.append(retrieved)
        all_relevant.append(relevant)
    
    # Measure efficiency
    print("\nMeasuring query latency...")
    latency_stats = benchmark.measure_query_latency(retrieval_system, queries)
    
    print("\nMeasuring index size...")
    size_stats = benchmark.measure_index_size(index_path)
    
    # Calculate effectiveness metrics (if relevance judgments available)
    effectiveness_metrics = {}
    if any(len(rel) > 0 for rel in all_relevant):
        print("\nCalculating effectiveness metrics...")
        effectiveness_metrics = evaluator.evaluate_queries(all_retrieved, all_relevant)
    
    # Generate report
    efficiency_metrics = {**latency_stats, **size_stats}
    
    output_path = args.output or os.path.join(config['paths']['results_dir'], 'evaluation.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = benchmark.generate_report(output_path, effectiveness_metrics, efficiency_metrics)
    
    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    print("\nEfficiency Metrics:")
    for metric, value in efficiency_metrics.items():
        if 'latency' in metric:
            print(f"  {metric}: {value*1000:.2f} ms")
        elif 'size_mb' in metric:
            print(f"  {metric}: {value:.2f} MB")
        else:
            print(f"  {metric}: {value}")
    
    if effectiveness_metrics:
        print("\nEffectiveness Metrics:")
        for metric, value in effectiveness_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nFull report saved to: {output_path}")
    print("=" * 80)


def demo(args, config):
    """Run a demo with sample data."""
    print("=" * 80)
    print("IR SYSTEM DEMO")
    print("=" * 80)
    
    # Create sample documents
    sample_docs = [
        {
            'doc_id': 'doc1',
            'text': 'Information retrieval is the process of obtaining information system resources that are relevant to an information need from a collection of those resources.'
        },
        {
            'doc_id': 'doc2',
            'text': 'Text mining, also referred to as text data mining, is the process of deriving high-quality information from text.'
        },
        {
            'doc_id': 'doc3',
            'text': 'Machine learning is a field of inquiry devoted to understanding and building methods that learn from data.'
        },
        {
            'doc_id': 'doc4',
            'text': 'Natural language processing is a subfield of linguistics, computer science, and artificial intelligence.'
        },
        {
            'doc_id': 'doc5',
            'text': 'Search engines use information retrieval techniques to find and rank documents based on user queries.'
        }
    ]
    
    print(f"\nCreated {len(sample_docs)} sample documents")
    
    # Preprocess
    print("\nPreprocessing documents...")
    preprocessor = TextPreprocessor(config['preprocessing'])
    processed_docs = [preprocessor.preprocess_document(d['doc_id'], d['text']) for d in sample_docs]
    
    # Build index
    print("Building index...")
    inverted_index = InvertedIndex()
    inverted_index.build(processed_docs)
    
    # Create retrieval system
    retrieval_system = RetrievalSystem(config, preprocessor, inverted_index, processed_docs)
    
    # Run sample queries
    sample_queries = [
        "information retrieval",
        "text mining",
        "machine learning"
    ]
    
    print("\n" + "=" * 80)
    print("DEMO QUERIES")
    print("=" * 80)
    
    for query in sample_queries:
        results, query_time = retrieval_system.search(query, top_k=3)
        print(format_results(results, query))
        print(f"Query time: {query_time*1000:.2f} ms\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Information Retrieval System - CS 516 Assignment 3',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build inverted index')
    index_parser.add_argument('--data-dir', required=True, help='Path to data directory or file')
    index_parser.add_argument('--output', help='Output directory for index')
    index_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('--index', help='Path to index directory')
    search_parser.add_argument('--query', help='Query string (for single query)')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate system performance')
    eval_parser.add_argument('--index', help='Path to index directory')
    eval_parser.add_argument('--queries', help='Path to queries JSON file')
    eval_parser.add_argument('--output', help='Output path for evaluation report')
    eval_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demo with sample data')
    demo_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Load config
    config = load_config(args.config)
    
    # Execute command
    if args.command == 'index':
        build_index(args, config)
    elif args.command == 'search':
        if args.query:
            search_single(args, config)
        else:
            search_interactive(args, config)
    elif args.command == 'evaluate':
        evaluate_system(args, config)
    elif args.command == 'demo':
        demo(args, config)


if __name__ == '__main__':
    main()
