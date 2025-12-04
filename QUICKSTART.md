# Quick Start Guide

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download NLTK Data** (automatic on first run, or manual):
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Quick Test

Run the demo to see the system in action:

```bash
python main.py demo
```

## Using Your Own Data

### 1. Prepare Your Documents

**Option A: Text File** (one document per line)
```
data/raw/documents.txt
```

**Option B: JSON File**
```json
[
  {"doc_id": "doc1", "text": "Your document text here..."},
  {"doc_id": "doc2", "text": "Another document..."}
]
```

### 2. Build Index

```bash
python main.py index --data-dir data/raw/documents.txt
```

### 3. Search

**Interactive**:
```bash
python main.py search --index models/index
```

**Single Query**:
```bash
python main.py search --index models/index --query "your query" --top-k 10
```

### 4. Evaluate

Create `results/queries.json`:
```json
[
  {"query": "test query", "relevant": ["doc1", "doc2"]}
]
```

Run evaluation:
```bash
python main.py evaluate --index models/index --queries results/queries.json
```

## Configuration

Edit `config.yaml` to customize:
- Preprocessing (stemming, stopwords)
- Ranking method (BM25 or TF-IDF)
- BM25 parameters (k1, b)
- Number of results (top_k)

## Troubleshooting

**Error: Module not found**
```bash
pip install -r requirements.txt
```

**Error: NLTK data not found**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**Error: Index not found**
```bash
python main.py index --data-dir data/raw
```

## File Structure

```
IR assignment/
├── src/              # Source code
├── data/raw/         # Your documents here
├── models/index/     # Generated indices
├── results/          # Evaluation results
├── main.py           # Main application
└── config.yaml       # Configuration
```

## For Submission

1. **Convert technical report to PDF**: `report/technical_report.md` → PDF
2. **Create GitHub repo**: Push all code
3. **Test with your dataset**: Verify everything works
4. **Submit**: PDF report + GitHub link
