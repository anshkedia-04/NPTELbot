# 🎓 NPTEL IoT Exam-Prep Assistant

An exam-preparation companion for the **NPTEL *Introduction to Internet of Things*** course (Weeks 1–12), combining an interactive mock test simulator, a pre-generated 60-lecture study guide, and a Groq-powered RAG AI tutor grounded in the official lecture transcripts.

---

## ✨ Features

### 🎯 Mock Test Simulator *(No API key required)*
- **180 official assignment MCQs** across Weeks 1–12.
- Choose **All Weeks (1–12)** or focus on a **specific week**, and pick **10, 25, or 50 questions**.
- Instant answer verification, official explanations, live accuracy tracking, and an end-of-test **Review Missed Questions** summary.

### 🧠 AI Tutor (RAG Chatbot)
- Grounded in all **60 official lecture transcripts** using FAISS + `sentence-transformers/all-MiniLM-L6-v2`.
- Powered by **Groq** (`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, and `openai/gpt-oss-120b`) for fast responses.
- **Four study modes:** `Q&A`, `Summarize Topic`, `Explain Simply`, and `Quiz Me`.
- **Lecture Scope filter:** Search across all 60 lectures or restrict answers to a single lecture.
- **Flexible API key support (BYOK):** Reads `GROQ_API_KEY` from `.env`, Streamlit Secrets, or the sidebar.

### 📖 Full Course Study Guide *(No API key required)*
- Detailed Markdown notes for all **60 lectures** (Weeks 1–12), naturally sorted.
- Instant **keyword search** and **week filter** for rapid revision.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | Streamlit |
| **LLM Orchestration** | LangChain (`langchain-groq`) |
| **Vector Store** | FAISS (`faiss-cpu`) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (`langchain-huggingface`) |
| **Deployment** | Streamlit Community Cloud |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/NPTEL_RAG_Bot.git
cd NPTEL_RAG_Bot

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure Groq API key for the AI Tutor tab
cp .env.example .env
# Edit .env and add GROQ_API_KEY
# (Or enter your key directly in the app sidebar)

# 5. Run the Streamlit app
streamlit run app.py
```

*(Pre-built `faiss_index_iot/`, `study_guide.json`, and `questions.json` are already included. Run `python build_vectorstore.py` or `python build_study_guide.py` only if you want to regenerate them from `nptel_transcripts/`.)*

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select your repository, branch (`main`), and main file path (`app.py`).
4. *(Optional)* In **App Settings → Secrets**, add your Groq key if you want the AI Tutor pre-configured for visitors:
   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```
   If omitted, visitors can still enter their own free key from [Groq Console](https://console.groq.com/keys) in the sidebar, and **Mock Test** + **Study Guide** work without any key.

---

## 📜 License & Attribution

- Course transcripts and assignment questions belong to **NPTEL** and the respective course instructors.
- Application code is licensed under the **MIT License** (see `LICENSE`).
