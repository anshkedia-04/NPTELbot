import os

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. Define Paths ---
TRANSCRIPT_FOLDER = "./nptel_transcripts/"
VECTORSTORE_PATH = "faiss_index_iot"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- 2. Load Documents with Metadata ---
print("Loading documents...")
all_docs = []
for filename in sorted(os.listdir(TRANSCRIPT_FOLDER)):
    if filename.endswith(".txt"):
        file_path = os.path.join(TRANSCRIPT_FOLDER, filename)
        loader = TextLoader(file_path, encoding="utf-8")
        doc = loader.load()[0]

        # "W1_L1_Intro.txt" -> "W1 L1 Intro"
        source_name = filename.replace(".txt", "").replace("_", " ")
        doc.metadata["source"] = source_name
        all_docs.append(doc)

print(f"Loaded {len(all_docs)} documents.")

# --- 3. Split Documents ---
print("Splitting documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = text_splitter.split_documents(all_docs)

# --- 4. Create Embeddings ---
print("Loading embedding model...")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"token": False})

# --- 5. Create and Save Vector Store ---
print("Creating and saving vector store...")
vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
vectorstore.save_local(VECTORSTORE_PATH)

print(f"\nDone! Vector store saved to {VECTORSTORE_PATH}")
