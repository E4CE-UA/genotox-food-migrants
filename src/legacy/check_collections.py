import chromadb

def list_collections():
    client = chromadb.Client()
    collections = client.list_collections()
    
    print("\nColecciones disponibles en ChromaDB:")
    if not collections:
        print("No hay colecciones disponibles.")
        return

    for i, collection in enumerate(collections):
        try:
            count = collection.count()
        except:
            count = "Error al contar documentos"
        
        print(f"{i+1}. {collection.name} (contiene {count} documentos)")
    
    print("\nPara usar una colección específica en el script de toxicidad:")
    print("1. Abre el archivo toxicidad.py")
    print("2. Busca la línea: CHROMADB_COLLECTION = \"chemicals_collection\"")
    print("3. Cambia 'chemicals_collection' por el nombre exacto de la colección que quieres usar")
    print("   Por ejemplo: CHROMADB_COLLECTION = \"[nombre_de_tu_colección]\"")

if __name__ == "__main__":
    list_collections()