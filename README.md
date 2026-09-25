# 🎓 Saksham Agent AI — Track 3: Transcript & Study Plan Agent

An autonomous AI agent built for **Track 3 (Education)** that proactively processes academic transcripts, identifies weak subject areas, generates an actionable study plan immediately upon ingestion, and provides an interactive study assistant powered by RAG (Retrieval-Augmented Generation).

---

## 🚀 Key Features

* **Proactive Document Processing:** Automatically scans uploaded PDF transcripts for weak grades (`Grade C`, `Grade D`, `Grade F`, or low score ratios) and generates a structured study plan on ingestion before any user prompt.
* **Retrieval-Augmented Generation (RAG):** Uses `ChromaDB` vector search and `HuggingFaceEmbeddings` to answer course-specific questions grounded directly in student document context.
* **Tool Routing:** Instantly delivers relevant developer and data analyst internship matches when job-related queries are asked.
* **Fast & Reliable Execution:** Powered by `Google Gemini 2.5/3.6 Flash` for fast, real-time responses during live demonstrations.

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit
* **LLM:** Google Gemini Flash via `langchain-google-genai`
* **Embeddings:** `BAAI/bge-small-en-v1.5` (`HuggingFaceEmbeddings`)
* **Vector Store:** ChromaDB
* **Document Parsing:** `PyPDFLoader` (`langchain-community`)

---

## 📦 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/pavanpoonamsaroj-ui/Saksham_agent_ai.git](https://github.com/pavanpoonamsaroj-ui/Saksham_agent_ai.git)
   cd Saksham_agent_ai
