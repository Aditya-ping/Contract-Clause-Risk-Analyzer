import json
import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import chromadb

# Global variables for models and vector store client
classifier_tokenizer = None
classifier_model = None
embedding_model = None
chroma_collection = None

# Standard category mapping fallback matching dataset label indices (v2: 9 categories)
LABEL_MAP = {
    0: "License Grant",
    1: "Cap On Liability",
    2: "Audit Rights",
    3: "Anti-Assignment",
    4: "Insurance",
    5: "Governing Law",
    6: "Post-Termination Services",
    7: "Minimum Commitment",
    8: "Exclusivity"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier_tokenizer, classifier_model, embedding_model, chroma_collection

    print("Initializing NDA Clause Risk Analyzer models and database...")

    # 1. Load fine-tuned DistilBERT classifier from HuggingFace Hub (v2)
    hf_model_id = "Adi2335/nda-clause-classifier-v2"
    print(f"Loading sequence classification model from HF Hub: {hf_model_id}")
    classifier_tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    classifier_model = AutoModelForSequenceClassification.from_pretrained(hf_model_id)
    classifier_model.eval()

    # 2. Load SentenceTransformer embedding model
    st_model_name = "all-MiniLM-L6-v2"
    print(f"Loading SentenceTransformer model: {st_model_name}")
    embedding_model = SentenceTransformer(st_model_name)

    # 3. Initialize ChromaDB PersistentClient and vector collection
    chroma_db_path = "./chroma_db"
    print(f"Initializing ChromaDB PersistentClient at {chroma_db_path}")
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    chroma_collection = chroma_client.get_or_create_collection(
        name="nda_clauses",
        metadata={"hnsw:space": "cosine"}
    )

    # 4. Check if collection is empty; if so, populate from corpus file
    existing_count = chroma_collection.count()
    print(f"Current ChromaDB collection document count: {existing_count}")

    if existing_count == 0:
        corpus_path = "nda_clauses_corpus.json"
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"Corpus file '{corpus_path}' not found for initial indexing.")

        print(f"Indexing corpus data from {corpus_path}...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)

        total_items = len(corpus_data)
        print(f"Loaded {total_items} clause items from corpus. Embedding and inserting into ChromaDB...")

        batch_size = 500
        for i in range(0, total_items, batch_size):
            batch = corpus_data[i:i + batch_size]
            batch_texts = [item["text"] for item in batch]
            batch_metadatas = [{"category": item.get("category", "Unknown")} for item in batch]
            batch_ids = [f"clause_{i + j}" for j in range(len(batch))]

            batch_embeddings = embedding_model.encode(batch_texts, show_progress_bar=False).tolist()

            chroma_collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            print(f"Indexed batch {i // batch_size + 1}/{(total_items + batch_size - 1) // batch_size} ({len(batch)} items)")

        print(f"Indexing complete. Total items in collection: {chroma_collection.count()}")
    else:
        print("Vector database already contains index data. Skipping re-embedding startup process.")

    print("FastAPI application startup complete. Ready for requests.")
    yield
    print("Shutting down NDA Clause Risk Analyzer application.")


app = FastAPI(
    title="NDA Clause Risk Analyzer API",
    description="Local FastAPI microservice that classifies NDA clauses into risk categories and retrieves top similar precedent clauses.",
    version="1.0.0",
    lifespan=lifespan
)


class ClauseRequest(BaseModel):
    text: str = Field(..., description="NDA clause text to analyze", example="Neither party may assign this Agreement without prior written consent.")


class Precedent(BaseModel):
    text: str
    category: str
    distance: float


class ClauseAnalysisResponse(BaseModel):
    predicted_category: str
    confidence: float
    similar_precedents: List[Precedent]


@app.post("/analyze-clause", response_model=ClauseAnalysisResponse)
async def analyze_clause(request: ClauseRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    # 1. Sequence Classification with DistilBERT model
    inputs = classifier_tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = classifier_model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    pred_idx = torch.argmax(probabilities).item()
    confidence = float(probabilities[pred_idx].item())

    # Map index to category name using model config id2label or fallback mapping
    id2label = getattr(classifier_model.config, "id2label", None)
    if id2label and isinstance(id2label, dict) and (pred_idx in id2label or str(pred_idx) in id2label):
        predicted_category = id2label.get(pred_idx, id2label.get(str(pred_idx)))
        if predicted_category.startswith("LABEL_"):
            predicted_category = LABEL_MAP.get(pred_idx, predicted_category)
    else:
        predicted_category = LABEL_MAP.get(pred_idx, f"Category_{pred_idx}")

    # 2. Vector Similarity Search with SentenceTransformer & ChromaDB
    query_embedding = embedding_model.encode([request.text], show_progress_bar=False).tolist()

    query_results = chroma_collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    similar_precedents = []
    if query_results and query_results.get("documents") and query_results["documents"][0]:
        docs = query_results["documents"][0]
        metas = query_results["metadatas"][0]
        distances = query_results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            similar_precedents.append(
                Precedent(
                    text=doc,
                    category=meta.get("category", "Unknown"),
                    distance=float(dist)
                )
            )

    return ClauseAnalysisResponse(
        predicted_category=predicted_category,
        confidence=confidence,
        similar_precedents=similar_precedents
    )
