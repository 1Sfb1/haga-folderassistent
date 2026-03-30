# 🏥 HAGA Folder RAG — Zoekmachine door Patiëntenfolders

Een RAG (Retrieval-Augmented Generation) systeem dat 1800+ patiëntenfolders van het HagaZiekenhuis doorzoekbaar maakt via een chat-interface.

## Architectuur

```
PDF Folders → Extractie → Chunking → Embeddings → ChromaDB
                                                      ↓
                                          Gebruiker → Query
                                                      ↓
                                              Vector Search → Top-K chunks
                                                      ↓
                                              LLM + Context → Antwoord
```

## Vereisten

- Python 3.10+
- GPU met CUDA (aanbevolen voor lokale embeddings/LLM)
- ~4GB VRAM voor embeddings, ~8GB+ voor lokaal LLM

## Installatie

```bash
# 1. Maak virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# of: venv\Scripts\activate  # Windows

# 2. Installeer dependencies
pip install -r requirements.txt

# 3. Kopieer config
cp .env.example .env
# Pas .env aan naar je voorkeuren (LLM provider, paden, etc.)
```

## Gebruik

### Stap 0: Folders downloaden (eenmalig)

Als je de folders nog niet lokaal hebt:

```bash
python scrape_folders.py --output ./folders/
```

Dit probeert automatisch alle ~1800 folders te downloaden van `folders.hagaziekenhuis.nl`.
Als de automatische detectie niet werkt:

```bash
# Brute-force scan (duurt ~30 min, maar vindt alles)
python scrape_folders.py --output ./folders/ --method bruteforce
```

### Stap 1: Folders inladen (eenmalig)

Zet al je PDF-folders in een map (bijv. `./folders/`) en draai:

```bash
python ingest.py --folder-dir ./folders/
```

Dit doet het volgende:
- Extraheert tekst uit alle PDFs (met PyMuPDF)
- Splitst teksten op in chunks van ~500 tokens met overlap
- Genereert embeddings (lokaal met sentence-transformers óf via OpenAI)
- Slaat alles op in een ChromaDB database

Optionele flags:
- `--chunk-size 500` — tokens per chunk (default: 500)
- `--chunk-overlap 50` — overlap tussen chunks (default: 50)
- `--batch-size 64` — batch size voor embeddings (default: 64)
- `--provider openai` — gebruik OpenAI embeddings i.p.v. lokaal

### Stap 2: Start de server

```bash
python server.py
```

Ga naar `http://localhost:8000` voor de chat-interface.

## LLM Configuratie

### Optie A: Lokaal met Ollama (gratis, privacy-vriendelijk)

```bash
# Installeer Ollama: https://ollama.ai
ollama pull llama3.1:8b    # of een ander model
```

In `.env`:
```
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
```

### Optie B: OpenAI API

In `.env`:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Technische Details

| Component | Keuze | Waarom |
|-----------|-------|--------|
| PDF Extractie | PyMuPDF (fitz) | Snel, betrouwbaar, geen OCR nodig voor digitale PDFs |
| Chunking | Recursive Character Splitter | Respecteert alinea-grenzen |
| Embeddings | `intfloat/multilingual-e5-large` | Top-tier voor Nederlands, draait op GPU |
| Vector Store | ChromaDB | Geen externe service nodig, persistent |
| LLM | Ollama / OpenAI | Flexibel, makkelijk te switchen |
| Backend | FastAPI | Async, snel, modern Python |
| Frontend | Vanilla HTML/CSS/JS | Geen build-stap nodig |
