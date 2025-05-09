# ingest_docs.py
import os
import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DOCUMENTS_DIR = "documents"
CHROMA_DB_PATH = "db/chroma_db"
COLLECTION_NAME = "voice_agent_docs"
# Use a multilingual model
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
# For intfloat/multilingual-e5-large, it's recommended to prefix queries with "query: " and passages with "passage: "
# However, for simplicity in this example, we'll omit it for passages during ingestion.
# You might add "passage: " prefix if you see performance benefits.

# --- Initialize ---
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Found existing collection: {COLLECTION_NAME}")
except Exception: # Replace with specific exception type chromadb.errors.CollectionNotFoundError once you know it
    print(f"Creating new collection: {COLLECTION_NAME}")
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"} # Using cosine similarity
    )

def get_text_chunks(text, chunk_size=256, chunk_overlap=32): # Simple chunking
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

# --- Process Documents ---
doc_id_counter = 0
for filename in os.listdir(DOCUMENTS_DIR):
    if filename.endswith((".txt", ".md")):
        filepath = os.path.join(DOCUMENTS_DIR, filename)
        print(f"Processing document: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks = get_text_chunks(content) # You might use more sophisticated chunking from LangChain/LlamaIndex
            
            if not chunks:
                print(f"No chunks generated for {filename}. Skipping.")
                continue

            print(f"Generating embeddings for {len(chunks)} chunks from {filename}...")
            # For multilingual-e5, prefixing passages with "passage: " is recommended for optimal performance
            # but for simplicity, we'll encode directly. For optimal results, consider adding "passage: ".
            chunk_embeddings = embedding_model.encode([f"passage: {chunk}" for chunk in chunks]).tolist()

            ids = [f"{filename}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

            collection.add(
                ids=ids,
                embeddings=chunk_embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            doc_id_counter += len(chunks)
            print(f"Added {len(chunks)} chunks from {filename} to ChromaDB.")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

print(f"\nIngestion complete. Total documents/chunks processed: {doc_id_counter}")
print(f"Vector DB contains {collection.count()} embeddings.")

# Example query to test
query_text = "What is the capital of India?"
query_embedding = embedding_model.encode(f"query: {query_text}").tolist() # Prefix query for e5 models
results = collection.query(query_embeddings=[query_embedding], n_results=2)
print("\nTest query results for:", query_text)
for i, doc in enumerate(results['documents'][0]):
    print(f"  Result {i+1}: {doc} (Distance: {results['distances'][0][i]})")