import chromadb
import os

# ChromaDB location: set CAS_CHROMA_PATH, or run from a directory holding ./chroma_db
chroma_path = os.environ.get("CAS_CHROMA_PATH", "chroma_db")
print(f"Buscando en: {chroma_path}")

client = chromadb.PersistentClient(path=chroma_path)

# Listar colecciones
collections = client.list_collections()
print(f"Colecciones encontradas: {collections}")

# Probar acceder a la colección específica
try:
    collection = client.get_collection("nist_retention_data")
    print(f"Colección 'nist_retention_data' encontrada con {collection.count()} documentos")
except Exception as e:
    print(f"Error al buscar 'nist_retention_data': {e}")