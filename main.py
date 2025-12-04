import argparse
import yaml
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocessor import ContentCleaner, read_data_files
from indexer import ReverseIndex, ProbabilisticRanker, VectorSpaceRanker
from retrieval import QueryProcessor, format_results
from evaluator import StatsCalculator, SystemProfiler


def load_config(config_path: str = 'config.yaml') -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_index(args, config):
    print("=" * 80)
    print("BUILDING INDEX")
    print("=" * 80)
    
    print(f"\nLoading documents from: {args.data_dir}")
    
    documents = []
    data_path = Path(args.data_dir)
    
    if data_path.is_file():
        documents = read_data_files(str(data_path))
    elif data_path.is_dir():
        for file_path in data_path.glob('*'):
            if file_path.suffix in ['.txt', '.json']:
                docs = read_data_files(str(file_path))
                documents.extend(docs)
    else:
        print(f"Error: {args.data_dir} not found!")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    print("\nPreprocessing documents...")
    cleaner = ContentCleaner(config['preprocessing'])
    
    processed_docs = []
    for doc in documents:
        processed_doc = cleaner.clean_text_data(doc['doc_id'], doc['text'])
        processed_docs.append(processed_doc)
    
    print(f"Preprocessed {len(processed_docs)} documents")
    
    print("\nBuilding inverted index...")
    reverse_index = ReverseIndex()
    reverse_index.create_structure(processed_docs)
    
    print(f"Index contains {len(reverse_index.index)} unique terms")
    print(f"Average document length: {reverse_index.avg_doc_length:.2f} tokens")
    
    output_dir = args.output or config['paths']['index_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, 'inverted_index.pkl')
    reverse_index.save(index_path)
    print(f"\nSaved inverted index to: {index_path}")
    
    docs_path = os.path.join(output_dir, 'processed_docs.pkl')
    import pickle
    with open(docs_path, 'wb') as f:
        pickle.dump(processed_docs, f)
    print(f"Saved processed documents to: {docs_path}")
    
    if config['indexing']['method'] in ['tfidf', 'hybrid']:
        print("\nBuilding TF-IDF model...")
        tfidf_ranker = VectorSpaceRanker(processed_docs)
        tfidf_path = os.path.join(output_dir, 'tfidf_model.pkl')
        tfidf_ranker.save(tfidf_path)
        print(f"Saved TF-IDF model to: {tfidf_path}")
    
    print("\n" + "=" * 80)
    print("INDEX BUILD COMPLETE")
    print("=" * 80)


def search_interactive(args, config):
    print("=" * 80)
    print("INTERACTIVE SEARCH")
    print("=" * 80)
    
    index_dir = args.index or config['paths']['index_dir']
    
    print(f"\nLoading index from: {index_dir}")
    index_path = os.path.join(index_dir, 'inverted_index.pkl')
    reverse_index = ReverseIndex.load(index_path)
    
    docs_path = os.path.join(index_dir, 'processed_docs.pkl')
    import pickle
    with open(docs_path, 'rb') as f:
        processed_docs = pickle.load(f)
    
    print(f"Loaded index with {len(reverse_index.index)} terms")
    print(f"Loaded {len(processed_docs)} documents")
    
    cleaner = ContentCleaner(config['preprocessing'])
    processor = QueryProcessor(config, cleaner, reverse_index, processed_docs)
    
    print("\n" + "=" * 80)
    print("Ready to search! (Type 'quit' to exit)")
    print("=" * 80)
    
    while True:
        query = input("\nEnter query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if not query:
            continue
        
        results, query_time = processor.execute_query(query, top_k=args.top_k)
        
        print(format_results(results, query))
        print(f"\nQuery time: {query_time*1000:.2f} ms")


def search_single(args, config):
    index_dir = args.index or config['paths']['index_dir']
    
    index_path = os.path.join(index_dir, 'inverted_index.pkl')
    reverse_index = ReverseIndex.load(index_path)
    
    docs_path = os.path.join(index_dir, 'processed_docs.pkl')
    import pickle
    with open(docs_path, 'rb') as f:
        processed_docs = pickle.load(f)
    
    cleaner = ContentCleaner(config['preprocessing'])
    processor = QueryProcessor(config, cleaner, reverse_index, processed_docs)
    
    results, query_time = processor.execute_query(args.query, top_k=args.top_k)
    
    print(format_results(results, args.query))
    print(f"\nQuery time: {query_time*1000:.2f} ms")


def evaluate_system(args, config):
    print("=" * 80)
    print("SYSTEM EVALUATION")
    print("=" * 80)
    
    index_dir = args.index or config['paths']['index_dir']
    
    print(f"\nLoading index from: {index_dir}")
    index_path = os.path.join(index_dir, 'inverted_index.pkl')
    reverse_index = ReverseIndex.load(index_path)
    
    docs_path = os.path.join(index_dir, 'processed_docs.pkl')
    import pickle
    with open(docs_path, 'rb') as f:
        processed_docs = pickle.load(f)
    
    cleaner = ContentCleaner(config['preprocessing'])
    processor = QueryProcessor(config, cleaner, reverse_index, processed_docs)
    
    if args.queries:
        print(f"\nLoading queries from: {args.queries}")
        with open(args.queries, 'r') as f:
            query_data = json.load(f)
    else:
        print("\nUsing sample queries (no query file provided)")
        query_data = [
            {"query": "information retrieval", "relevant": []},
            {"query": "text mining", "relevant": []},
            {"query": "machine learning", "relevant": []}
        ]
    
    calculator = StatsCalculator(config)
    profiler = SystemProfiler()
    
    print(f"\nRunning {len(query_data)} queries...")
    all_retrieved = []
    all_relevant = []
    queries = []
    
    for item in query_data:
        query = item['query']
        relevant = item.get('relevant', [])
        
        queries.append(query)
        results, _ = processor.execute_query(query)
        retrieved = [r['doc_id'] for r in results]
        
        all_retrieved.append(retrieved)
        all_relevant.append(relevant)
    
    print("\nMeasuring query latency...")
    latency_stats = profiler.check_speed(processor, queries)
    
    print("\nMeasuring index size...")
    size_stats = profiler.check_storage(index_path)
    
    effectiveness_metrics = {}
    if any(len(rel) > 0 for rel in all_relevant):
        print("\nCalculating effectiveness metrics...")
        effectiveness_metrics = calculator.assess_batch(all_retrieved, all_relevant)
    
    efficiency_metrics = {**latency_stats, **size_stats}
    
    output_path = args.output or os.path.join(config['paths']['results_dir'], 'evaluation.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    report = profiler.generate_report(output_path, effectiveness_metrics, efficiency_metrics)
    
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
    print("=" * 80)
    print("IR SYSTEM DEMO")
    print("=" * 80)
    
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
    
    print("\nPreprocessing documents...")
    cleaner = ContentCleaner(config['preprocessing'])
    processed_docs = [cleaner.clean_text_data(d['doc_id'], d['text']) for d in sample_docs]
    
    print("Building index...")
    reverse_index = ReverseIndex()
    reverse_index.create_structure(processed_docs)
    
    processor = QueryProcessor(config, cleaner, reverse_index, processed_docs)
    
    sample_queries = [
        "information retrieval",
        "text mining",
        "machine learning"
    ]
    
    print("\n" + "=" * 80)
    print("DEMO QUERIES")
    print("=" * 80)
    
    for query in sample_queries:
        results, query_time = processor.execute_query(query, top_k=3)
        print(format_results(results, query))
        print(f"Query time: {query_time*1000:.2f} ms\n")


def main():
    parser = argparse.ArgumentParser(
        description='Information Retrieval System - CS 516 Assignment 3',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    index_parser = subparsers.add_parser('index', help='Build inverted index')
    index_parser.add_argument('--data-dir', required=True, help='Path to data directory or file')
    index_parser.add_argument('--output', help='Output directory for index')
    index_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('--index', help='Path to index directory')
    search_parser.add_argument('--query', help='Query string (for single query)')
    search_parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    search_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate system performance')
    eval_parser.add_argument('--index', help='Path to index directory')
    eval_parser.add_argument('--queries', help='Path to queries JSON file')
    eval_parser.add_argument('--output', help='Output path for evaluation report')
    eval_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    demo_parser = subparsers.add_parser('demo', help='Run demo with sample data')
    demo_parser.add_argument('--config', default='config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config = load_config(args.config)
    
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
