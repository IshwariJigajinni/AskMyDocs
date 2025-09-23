# 📄 AskMyDocs - GenAI Document Intelligence Platform

**Not a doctor, just a document whisperer.** AskMyDocs reads your PDFs so you don’t have to. 😎

Upload PDFs and ask questions with powerful retrieval-augmented generation (RAG) strategies. Built with **FastAPI**, **FAISS**, **OpenAI LLMs**, and **Streamlit** for a fully interactive experience.

---

## 🔹 Features

- Upload multiple PDFs and index them for search.
- Choose RAG strategies:
  - **Vanilla:** Basic similarity search.
  - **Reranking:** LLM scores retrieved chunks for relevance.
  - **HyDE:** Generates a hypothetical answer for better retrieval.
- Detailed retrieval metrics:
  - Latency
  - Chunks retrieved
  - Recall@k
  - Average & Median similarity
  - Query coverage (%)
- Source chunk inspection with expandable previews.
- Persistent vector storage — indexed PDFs survive server restarts.
- Config-driven setup (`config.yaml`) for easy experimentation.

---

## 🔹 Tech Stack

- **Backend:** FastAPI + OpenAI API + FAISS
- **Frontend:** Streamlit + Pandas
- **PDF Processing:** PyMuPDF
- **Environment Management:** python-dotenv

---

## 🔹 Installation

**1. Clone the repository:**

```bash
git clone https://github.com/IshwariJigajinni/AskMyDocs.git 
cd genai-document-intelligence
```
**2. Create and activate a virtual environment:**
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```


**4. Set your OpenAI API key:**
```bash
# Linux/macOS
export OPENAI_API_KEY="your_api_key"
# Windows
setx OPENAI_API_KEY "your_api_key"
```

**5. Run the backend:**
```bash
uvicorn backend.main:app --reload
```


**6. Run the frontend:**
```bash
streamlit run frontend/app.py
```