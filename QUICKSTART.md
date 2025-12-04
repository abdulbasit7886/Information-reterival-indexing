# Quick Start

## Setup
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Run Demo
```bash
python main.py demo
```

## Usage

### 1. Index
```bash
python main.py index --data-dir data/raw
```

### 2. Search
```bash
python main.py search --index models/index
```

### 3. Evaluate
```bash
python main.py evaluate --index models/index --queries results/queries.json
```
