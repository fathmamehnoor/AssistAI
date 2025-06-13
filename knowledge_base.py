import chromadb
import os
import torch
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB with persistent storage
chroma_client = chromadb.PersistentClient(path="./chromadb")
collection = chroma_client.get_or_create_collection(name="support_knowledge")

# Use SentenceTransformers embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

print(f"Model is using device: {embedding_model.device}")
# Load data text files
def load_data():
    data_dir = "data" 
    files = ["products.txt", "policies.txt", "qa.txt"]

    for file in files:
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            vector = embedding_model.encode(content).tolist()
            collection.add(documents=[content], ids=[file], embeddings=[vector])


if __name__ == "__main__":
    load_data()
    print("Knowledge base loaded successfully.")
