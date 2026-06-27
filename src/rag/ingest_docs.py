import os
import glob
from pathlib import Path

# Optional imports handled gracefully
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None

def chunk_text(text, chunk_size=500, chunk_overlap=100):
    """
    Splits text into overlapping chunks of a rough character length.
    Ensures sentence boundaries are preserved where possible.
    """
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(text[start:])
            break
            
        # Try to find a space or period to split cleanly
        split_point = text.find(" ", end - 20, end + 20)
        if split_point == -1:
            split_point = end
            
        chunks.append(text[start:split_point])
        start = split_point - chunk_overlap
        
    return chunks

def load_documents(docs_dir):
    """
    Scans docs_dir for markdown (.md) and text (.txt) files.
    Reads and returns a list of dictionaries with 'content' and 'metadata'.
    """
    documents = []
    path = Path(docs_dir)
    
    # Create sample docs directory if empty
    os.makedirs(path, exist_ok=True)
    
    # Write a sample README if none exist for quick testing
    readme_path = path / "paniit_hackathon.md"
    if not any(path.glob("*.*")) and not readme_path.exists():
        sample_doc = (
            "# PanIIT Hackathon 2026 - P2P Disaster Portal\n\n"
            "Abhinav and team built a Peer-to-Peer (P2P) Offline-First Disaster Collaboration Portal.\n"
            "The stack comprises FastAPI, SQLite, and a custom P2P mesh network synchronization layer.\n"
            "The system is designed for rescue workers operating in regions without cellular connectivity.\n"
            "State is synchronized using a custom gossip protocol that distributes small JSON diffs.\n"
            "We won the regional award for hardware optimization and practical feasibility."
        )
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(sample_doc)
        print(f"Created a sample project document at {readme_path} for testing.")

    # Find txt and md files
    files = glob.glob(str(path / "*.txt")) + glob.glob(str(path / "*.md"))
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append({
                    "content": content,
                    "metadata": {"source": file_name}
                })
        except Exception as e:
            print(f"Failed to read {file_name}: {e}")
            
    print(f"Loaded {len(documents)} document(s) from {docs_dir}")
    return documents

def build_vector_store(docs_dir, db_dir, collection_name="abhinav_portfolio"):
    """
    Main ingestion pipeline: chunks documents, embeddings, and stores in ChromaDB.
    """
    print("--- Starting RAG Ingestion ---")
    
    if chromadb is None:
        print("Error: 'chromadb' library is not installed.")
        print("Please install dependencies: pip install chromadb sentence-transformers")
        return None

    # 1. Load files
    docs = load_documents(docs_dir)
    if not docs:
        print("No documents to index.")
        return None
        
    # 2. Chunk files
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    chunk_counter = 0
    for doc in docs:
        chunks = chunk_text(doc["content"])
        source = doc["metadata"]["source"]
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({"source": source, "chunk_index": i})
            all_ids.append(f"{source}_chunk_{i}")
            chunk_counter += 1
            
    print(f"Split documents into {chunk_counter} overlapping chunks.")

    # 3. Initialize ChromaDB client
    os.makedirs(db_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=db_dir)
    
    # 4. Initialize embedding function (uses local HF download)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # 5. Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=emb_fn
    )
    
    # 6. Add documents to store (upsert avoids duplicate errors)
    collection.upsert(
        ids=all_ids,
        documents=all_chunks,
        metadatas=all_metadatas
    )
    
    print(f"Successfully indexed {chunk_counter} chunks into local ChromaDB collection: '{collection_name}'")
    print("RAG document ingestion complete!\n")
    return collection

if __name__ == "__main__":
    # Default paths
    src_docs = "data/rag_docs"
    vector_db = "data/vector_store"
    
    build_vector_store(src_docs, vector_db)
