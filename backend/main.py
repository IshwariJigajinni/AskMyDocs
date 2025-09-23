from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os, time
from backend.vectorstore.store import VectorStore
from backend.utils import parse_pdf, chunk_text
from backend.gpt import answer_question, llm_generate, get_embedding
from typing import List
import yaml

app = FastAPI()
vector_store = VectorStore()

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

UPLOAD_FOLDER = "documents/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

TOP_K = config["top_k"]

# ---------------- Upload ----------------
@app.post("/upload")
async def upload_document(files: List[UploadFile] = File(...)):
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        text = parse_pdf(file_path)
        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            vector_store.add_text(chunk, metadata={"filename": file.filename, "chunk_id": idx})

    return {"message": f"{len(files)} files uploaded and indexed successfully."}

# ---------------- Query ----------------
class QueryRequest(BaseModel):
    query: str
    llm_model: str
    strategy: str

@app.post("/query")
async def query_document(request: QueryRequest):
    start_time = time.time()
    user_query = request.query
    llm_model = request.llm_model
    strategy = request.strategy.lower()
    top_k = 5
    results = vector_store.similarity_search(user_query, top_k=TOP_K)

    # ---------------- RAG Strategies ----------------
    if strategy == "vanilla":
        print("\n--- Vanilla RAG ---")
        print(f"Question: {user_query}")
        print("Retrieving chunks with plain similarity search...")
        results = vector_store.similarity_search(user_query, top_k=top_k)
        print(f"Retrieved {len(results)} chunks.\n")
        print("---------------------------\n")

    elif strategy == "reranking":
        reranked = []
        print("\n--- Reranking QA Debug ---")
        print(f"Question: {user_query}\n")
        for idx, chunk in enumerate(results, start=1):
            prompt = f"""
            Question: {user_query}
            Chunk: {chunk['text']}
            Score the relevance of this chunk to the question on a scale of 1-10.
            Return only the number.
            """
            score_text = llm_generate(prompt, llm_model=llm_model)
            try:
                score = int(score_text.strip())
            except:
                score = 0
            chunk["rerank_score"] = score
            reranked.append(chunk)

            print(f"Chunk {idx}: {chunk['metadata'].get('filename','?')} (chunk_id={chunk['metadata'].get('chunk_id','?')})")
            print(f"Relevance Score: {score}")
            print(f"Text (first 200 chars): {chunk['text'][:200]}...\n")

        results = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
         # Print sorted order safely
        if results:
            print("Sorted chunks after reranking:")
            for idx, chunk in enumerate(results, start=1):
                print(f"{idx}. Chunk_id={chunk['metadata'].get('chunk_id','?')}, Score={chunk['rerank_score']}")
        else:
            print("No chunks retrieved for reranking.")
        print("---------------------------\n")

    elif strategy == "hyde":
        hypo_prompt = f"Generate a detailed hypothetical answer to the question:\n{user_query}"
        hypothetical_answer = llm_generate(hypo_prompt, llm_model=llm_model)
        print("\n--- HyDE Hypothetical QA ---")
        print(f"Question: {user_query}")
        print(f"Hypothetical Answer (used for retrieval): {hypothetical_answer}")
        print("---------------------------\n")
        hypo_embedding = get_embedding(hypothetical_answer)
        results = vector_store.similarity_search_by_vector(hypo_embedding, top_k=top_k)

        # ---------------- Metrics ----------------
    chunks_retrieved = len(results)
    similarity_scores = [round(chunk.get("score", 0), 3) for chunk in results]

    # Keyword-based recall (improved)
    query_keywords = set(user_query.lower().split())
    retrieved_text = " ".join([chunk["text"].lower() for chunk in results])
    matched_keywords = [kw for kw in query_keywords if kw in retrieved_text]
    recall_at_k = len(matched_keywords) / max(1, len(query_keywords))  # fraction matched
    query_coverage = round(recall_at_k * 100, 2)  # percentage coverage

    # Similarity statistics
    avg_similarity = round(sum(similarity_scores) / max(1, len(similarity_scores)), 3)
    median_similarity = round(sorted(similarity_scores)[len(similarity_scores)//2], 3) if similarity_scores else 0

    # ---------------- Answer ----------------
    answer = answer_question(user_query, results, llm_model=llm_model)
    latency = round(time.time() - start_time, 2)

    return {
        "answer": answer,
        "sources": results,
        "llm_model": llm_model,
        "strategy": strategy,
        "metrics": {
            "latency": latency,
            "chunks_retrieved": chunks_retrieved,
            "similarity_scores": similarity_scores,
            "avg_similarity": avg_similarity,
            "median_similarity": median_similarity,
            "recall_at_k": recall_at_k,
            "query_coverage": query_coverage
        }
    }
