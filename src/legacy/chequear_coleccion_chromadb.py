import chromadb
import pandas as pd
import os

# ChromaDB location: set CAS_CHROMA_PATH, or run from a directory holding ./chroma_db
def examine_chemicals_collection(chroma_path=os.environ.get("CAS_CHROMA_PATH", "chroma_db")):
    """
    Examina específicamente la colección 'chemicals_collection' para contar CAS/CID
    """
    print(f"Conectando a ChromaDB en {chroma_path}...")
    
    try:
        # Conectar a ChromaDB
        client = chromadb.PersistentClient(path=chroma_path)
        
        print("\nObteniendo información de 'chemicals_collection'...")
        collection = client.get_collection("chemicals_collection")
        
        # Contar total de documentos
        # Usar paginación para obtener todos los IDs
        all_ids = []
        all_metadatas = []
        batch_size = 1000
        offset = 0
        
        while True:
            results = collection.get(limit=batch_size, offset=offset, include=["metadatas"])
            batch_ids = results["ids"]
            
            if not batch_ids:
                break
                
            all_ids.extend(batch_ids)
            all_metadatas.extend(results["metadatas"])
            offset += batch_size
            print(f"Recuperados {len(all_ids)} documentos hasta ahora...")
            
            # Romper el bucle después de la primera iteración para pruebas rápidas
            # (comentar estas líneas para un conteo completo)
            # if offset > 0:
            #    break
        
        total_docs = len(all_ids)
        print(f"\nTotal de documentos en 'chemicals_collection': {total_docs}")
        
        # Contar cuántos documentos tienen CAS y CID
        cas_count = 0
        cid_count = 0
        both_count = 0
        
        # Verificar primeros metadatos para ver qué campos tienen
        if all_metadatas:
            first_metadata = all_metadatas[0]
            print(f"\nEjemplo de metadatos: {first_metadata}")
            print(f"Campos disponibles: {', '.join(first_metadata.keys())}")
        
        # Contar documentos con CAS/CID
        for metadata in all_metadatas:
            has_cas = 'cas' in metadata and metadata['cas'] is not None and metadata['cas'] != ''
            has_cid = 'cid' in metadata and metadata['cid'] is not None and metadata['cid'] != ''
            
            if has_cas:
                cas_count += 1
            if has_cid:
                cid_count += 1
            if has_cas and has_cid:
                both_count += 1
        
        print(f"\nEstadísticas:")
        print(f"- Documentos con CAS: {cas_count} ({cas_count/total_docs*100:.2f}%)")
        print(f"- Documentos con CID: {cid_count} ({cid_count/total_docs*100:.2f}%)")
        print(f"- Documentos con ambos CAS y CID: {both_count} ({both_count/total_docs*100:.2f}%)")
        
        # Exportar una muestra a Excel para análisis
        sample_size = min(100, total_docs)
        print(f"\nExportando muestra de {sample_size} documentos a Excel para análisis...")
        
        # Convertir metadatos a DataFrame
        sample_data = all_metadatas[:sample_size]
        df = pd.DataFrame(sample_data)
        
        # Guardar a Excel
        sample_file = "chemicals_collection_sample.xlsx"
        df.to_excel(sample_file, index=False)
        print(f"Muestra guardada en {os.path.abspath(sample_file)}")
        
        # Preguntar si quiere exportar todos los CAS a un archivo
        export_all = input("\n¿Quieres exportar todos los CAS a un archivo? (s/n, default: n): ").strip().lower()
        if export_all == 's':
            print("Exportando todos los CAS...")
            
            # Extraer CAS de todos los metadatos
            all_cas = []
            for metadata in all_metadatas:
                if 'cas' in metadata and metadata['cas']:
                    all_cas.append({'CAS': metadata['cas']})
            
            # Guardar a Excel
            cas_file = "all_cas_numbers.xlsx"
            cas_df = pd.DataFrame(all_cas)
            cas_df.to_excel(cas_file, index=False)
            print(f"CAS guardados en {os.path.abspath(cas_file)}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    examine_chemicals_collection()