# NDA Clause Risk Analyzer

A local FastAPI microservice for NDA contract clause classification and similarity precedent retrieval.

## Features
- **Clause Classification**: Classifies text into 9 NDA clause categories using the fine-tuned DistilBERT model [`Adi2335/nda-clause-classifier-v2`](https://huggingface.co/Adi2335/nda-clause-classifier-v2).
- **Precedent Retrieval**: Embeds clauses using `all-MiniLM-L6-v2` via `sentence-transformers` and queries top-3 similar precedent clauses stored in a persistent `chromadb` vector index (`./chroma_db`).
- **Swagger Documentation**: Automated API documentation and interactive testing interface at `/docs`.

---

## 9 NDA Clause Categories
1. License Grant
2. Cap On Liability
3. Audit Rights
4. Anti-Assignment
5. Insurance
6. Governing Law
7. Post-Termination Services
8. Minimum Commitment
9. Exclusivity

---

## Prerequisites
- Python 3.9+ installed on your system.

---

## Setup & Installation Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Service

Launch the FastAPI application using Uvicorn:
```bash
uvicorn main:app --reload
```

On first startup:
- Downloads `Adi2335/nda-clause-classifier` from HuggingFace Hub.
- Downloads `all-MiniLM-L6-v2` embedding model.
- Indexes ~6,115 clause objects from `nda_clauses_corpus.json` into `./chroma_db`.
- On subsequent runs, ChromaDB automatically reuses the persisted vector database and skips re-embedding.

---

## API Interface

Open your browser and navigate to:
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Endpoint: `POST /analyze-clause`

#### Example Request Body
```json
{
  "text": "Neither party may assign or transfer this Agreement or any rights hereunder without prior written consent."
}
```

#### Example Response
```json
{
  "predicted_category": "Anti-Assignment",
  "confidence": 0.9984,
  "similar_precedents": [
    {
      "text": "No assignment of this Agreement or any right accruing hereunder shall be made by the Distributor in whole or in part, without the prior written consent of the Company.",
      "category": "Anti-Assignment",
      "distance": 0.1428
    },
    {
      "text": "Neither this Agreement nor any right hereunder or interest herein may be assigned or transferred by Consultant without the express written consent of Company.",
      "category": "Anti-Assignment",
      "distance": 0.1652
    },
    {
      "text": "Consultant shall not subcontract any portion of Consultant's duties under this Agreement without the prior written consent of Company.",
      "category": "Anti-Assignment",
      "distance": 0.2104
    }
  ]
}
```
