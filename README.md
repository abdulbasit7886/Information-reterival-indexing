# Information Retrieval System - CS 516 Assignment 3

A complete local information retrieval system implementing BM25 and TF-IDF ranking algorithms with comprehensive preprocessing and evaluation capabilities.

## 📋 Table of Contents

- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Multiple Ranking Algorithms**: BM25 (default) and TF-IDF with cosine similarity
- **Comprehensive Preprocessing**: Tokenization, stemming, stopword removal, normalization
- **Inverted Index**: Efficient document retrieval with posting lists
- **Evaluation Metrics**: Precision@K, Recall@K, MAP, NDCG
- **Performance Benchmarking**: Query latency, index size, throughput analysis
- **Command-Line Interface**: Easy-to-use CLI for indexing, searching, and evaluation
- **Fully Local**: No cloud dependencies, runs entirely on your machine

## 🖥️ System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended for large datasets)
- **Storage**: Varies by dataset size (index is ~10-20% of original data size)

## 📦 Installation

### Step 1: Clone or Download the Repository

```bash
cd "c:\Users\Abdul Basit\Desktop\IR assignment"
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

The system will automatically download required NLTK data on first run, but you can pre-download it:

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 🚀 Quick Start

### Run the Demo

The fastest way to see the system in action:

```bash
python main.py demo
```

This will create sample documents and run example queries.

### Index Your Documents

Place your documents in the `data/raw` directory, then:

```bash
# For a single file
python main.py index --data-dir data/raw/documents.txt

# For a directory of files
python main.py index --data-dir data/raw
```

### Search

```bash
# Interactive search
python main.py search --index models/index

# Single query
python main.py search --index models/index --query "information retrieval" --top-k 10
```

## 📖 Usage

### Building an Index

```bash
python main.py index --data-dir <path-to-documents> [--output <index-directory>]
```

**Supported Formats**:
- **TXT**: One document per line
- **JSON**: Array of objects with `doc_id` and `text` fields, or dictionary mapping doc_id to text

**Example JSON format**:
```json
[
  {"doc_id": "doc1", "text": "Information retrieval is..."},
  {"doc_id": "doc2", "text": "Text mining extracts..."}
]
```

### Searching

**Interactive Mode**:
```bash
python main.py search --index models/index
```

**Single Query**:
```bash
python main.py search --index models/index --query "your query here" --top-k 20
```

### Evaluation

```bash
python main.py evaluate --index models/index --queries results/queries.json --output results/evaluation.json
```

**Query File Format** (`queries.json`):
```json
[
  {
    "query": "information retrieval",
    "relevant": ["doc1", "doc3", "doc5"]
  },
  {
    "query": "text mining",
    "relevant": ["doc2", "doc7"]
  }
]
```

## 📁 Project Structure

```
IR-Assignment/
├── src/                          # Source code
│   ├── __init__.py              # Package initialization
│   ├── preprocessor.py          # Text preprocessing
│   ├── indexer.py               # Inverted index & ranking
│   ├── retrieval.py             # Query processing
│   └── evaluator.py             # Evaluation metrics
├── data/                         # Data directory
│   ├── raw/                     # Original documents
│   └── processed/               # Preprocessed documents
├── models/                       # Saved models
│   └── index/                   # Inverted index files
├── results/                      # Evaluation results
├── report/                       # Technical report
├── main.py                       # Main CLI application
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## ⚙️ Configuration

Edit `config.yaml` to customize system behavior:

### Preprocessing Options

```yaml
preprocessing:
  lowercase: true                 # Convert to lowercase
  remove_punctuation: true        # Remove punctuation
  remove_stopwords: true          # Filter stopwords
  stemming: porter                # porter, snowball, lemmatize, none
  min_token_length: 2             # Minimum token length
  max_token_length: 50            # Maximum token length
```

### Indexing Options

```yaml
indexing:
  method: bm25                    # bm25, tfidf, hybrid
  bm25_k1: 1.5                    # BM25 term frequency saturation
  bm25_b: 0.75                    # BM25 length normalization
  tfidf_norm: l2                  # TF-IDF normalization (l1, l2, none)
```

### Retrieval Options

```yaml
retrieval:
  top_k: 10                       # Default number of results
  min_score: 0.0                  # Minimum relevance score
  query_expansion: false          # Enable pseudo-relevance feedback
```

### Evaluation Options

```yaml
evaluation:
  metrics: [precision, recall, map, ndcg]
  k_values: [5, 10, 20]           # K values for P@K, R@K, NDCG@K
  benchmark_queries: 20           # Number of queries for benchmarking
```

## 📊 Evaluation

The system provides comprehensive evaluation metrics:

### Effectiveness Metrics

- **Precision@K**: Proportion of relevant documents in top-K results
- **Recall@K**: Proportion of relevant documents retrieved in top-K
- **MAP (Mean Average Precision)**: Average precision across all queries
- **NDCG@K**: Normalized Discounted Cumulative Gain at K

### Efficiency Metrics

- **Query Latency**: Mean, median, P95, P99 response times
- **Index Size**: Memory footprint in MB
- **Throughput**: Queries per second

### Running Evaluation

```bash
python main.py evaluate --index models/index --queries results/queries.json
```

Results are saved to `results/evaluation.json`:

```json
{
  "effectiveness_metrics": {
    "precision@5": 0.8500,
    "recall@10": 0.7200,
    "MAP": 0.7850,
    "ndcg@10": 0.8120
  },
  "efficiency_metrics": {
    "mean_latency": 0.0045,
    "p95_latency": 0.0089,
    "index_size_mb": 15.3
  }
}
```

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'nltk'"

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "LookupError: Resource punkt not found"

**Solution**: Download NLTK data:
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Issue: Index file not found

**Solution**: Build the index first:
```bash
python main.py index --data-dir data/raw
```

### Issue: Out of memory during indexing

**Solution**: 
- Process documents in batches
- Reduce `max_token_length` in config
- Use a machine with more RAM

### Issue: Slow query performance

**Solution**:
- Reduce `top_k` value
- Use BM25 instead of TF-IDF (faster)
- Ensure index is properly saved and loaded

## 📝 Citation

If you use this system in your research, please cite:

```
@misc{ir_system_2025,
  author = {CS 516 Student},
  title = {Information Retrieval System},
  year = {2025},
  publisher = {Information Technology University},
  course = {CS 516: Information Retrieval and Text Mining}
}
```

## 📄 License

This project is created for educational purposes as part of CS 516 coursework at ITU.

## 👤 Author

**CS 516 Student**  
Information Technology University (ITU)  
Fall 2025

## 🙏 Acknowledgments

- Dr. Ahmad Mustafa (Course Instructor)
- NLTK Development Team
- scikit-learn Contributors
- rank-bm25 Library Authors

---

**Note**: This system is designed for educational purposes and demonstrates core IR concepts. For production use, consider additional optimizations and features.
#   I n f o r m a t i o n - r e t e r i v a l - i n d e x i n g  
 