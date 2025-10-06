# NASA RAG (Chroma + TinyLlama)

## Setup
```
git clone <this-repo>
cd nasa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Build the Chroma index
```
python scripts/build_index.py --csv data/papers.csv --db ./chroma_nasa
```

## Run the Server
```
python scripts/serve.py
# -> server at http://127.0.0.1:7895
```

## Test without a frontend
```
python scripts/client_demo.py 
``` 
