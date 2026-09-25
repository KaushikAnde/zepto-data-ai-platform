
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"

# Embedding model required for the project
model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent ChromaDB
client = chromadb.PersistentClient(path=str(DB_DIR))

collection = client.get_or_create_collection(
    name="zepto_policies",
    metadata={"hnsw:space": "cosine"}
)


def chunk_text(text, chunk_size=500, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


documents = []
ids = []
metadatas = []

doc_files = sorted(DOCS_DIR.glob("*.txt"))

for file in doc_files:
    text = file.read_text(encoding="utf-8")
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        ids.append(f"{file.stem}_chunk_{i}")
        metadatas.append({
            "source": file.name,
            "chunk": i
        })


embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

print("Documents found:", len(doc_files))
print("Chunks created:", len(documents))
print("Chunks stored in ChromaDB:", collection.count())
print("ChromaDB path:", DB_DIR)
