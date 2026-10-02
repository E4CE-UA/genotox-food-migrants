import chromadb
import pandas as pd
from bs4 import BeautifulSoup
import re
import os

def clean_compound_name(name):
    """Limpia el nombre del compuesto de artefactos HTML"""
    if not name:
        return 'Unknown Compound'
    
    # Limpia artefactos comunes
    name = name.replace('x0028', '(')
    name = name.replace('_x0029_', ')')
    name = re.sub(r'[*_]x[0-9a-fA-F]+', '', name)
    name = ' '.join(name.split())
    
    return name.strip()

def get_all_documents(collection, batch_size=1000):
    """
    Obtiene todos los documentos de la colección usando paginación
    
    Args:
        collection: Colección ChromaDB
        batch_size: Tamaño de cada lote a recuperar
        
    Returns:
        dict: Diccionario con 'ids', 'metadatas' y 'documents'
    """
    all_results = {"ids": [], "metadatas": [], "documents": []}
    
    # Primero obtener conteo total
    # Como no hay método directo para contar, obtenemos solo los IDs
    count_results = collection.get(include=["metadatas"])
    total_count = len(count_results['ids'])
    
    print(f"Total de documentos en la colección: {total_count}")
    
    # Procesar en lotes
    offset = 0
    while offset < total_count:
        print(f"Obteniendo lote {offset//batch_size + 1} (documentos {offset}-{min(offset+batch_size-1, total_count-1)})...")
        
        # Obtener lote
        batch_results = collection.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas", "documents"]
        )
        
        # Añadir resultados a la lista completa
        all_results["ids"].extend(batch_results["ids"])
        all_results["metadatas"].extend(batch_results["metadatas"])
        all_results["documents"].extend(batch_results["documents"])
        
        # Actualizar offset
        offset += batch_size
        
        print(f"Procesados {len(all_results['ids'])} documentos de {total_count}")
    
    return all_results

# ChromaDB location: set CAS_CHROMA_PATH, or run from a directory holding ./chroma_db
def parse_retention_tables(chroma_path=os.environ.get("CAS_CHROMA_PATH", "chroma_db"), output_file="retention_index_data.xlsx"):
    """
    Parsea todas las tablas de retention index desde ChromaDB y las guarda en Excel
    
    Args:
        chroma_path (str): Ruta a la base de datos ChromaDB
        output_file (str): Nombre del archivo Excel de salida
    """
    print(f"Conectando a ChromaDB en {chroma_path}...")
    
    # Conectar a ChromaDB
    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collection = client.get_collection("nist_retention_data")
        print("Conexión exitosa a la colección 'nist_retention_data'")
    except Exception as e:
        print(f"Error al conectar a ChromaDB: {e}")
        return None
    
    # Obtener todos los documentos usando paginación
    print("Obteniendo documentos de ChromaDB con paginación...")
    results = get_all_documents(collection)
    
    if not results['documents'] or len(results['documents']) == 0:
        print("No se encontraron documentos en la colección.")
        return None
    
    print(f"Se encontraron {len(results['documents'])} documentos. Procesando...")
    
    # Lista para almacenar todos los datos
    all_data = []
    processed_count = 0
    error_count = 0
    
    # Procesar cada documento
    for i, (metadata, document) in enumerate(zip(results['metadatas'], results['documents'])):
        try:
            # Obtener información del compuesto
            cas_number = metadata.get('cas', 'N/A')
            substance_name = metadata.get('substance_name', 'Unknown')
            nist_name = metadata.get('nist_name', substance_name)
            
            # Limpiar nombre
            clean_name = clean_compound_name(substance_name)
            
            # Mostrar progreso cada 10 documentos
            if i % 10 == 0:
                print(f"[{i+1}/{len(results['documents'])}] Procesando {clean_name} (CAS: {cas_number})")
            
            # Parsear el HTML
            soup = BeautifulSoup(document, 'html.parser')
            
            # Encontrar todas las tablas
            tables = soup.find_all('table')
            
            if not tables:
                if i % 10 == 0:
                    print(f"  No se encontraron tablas para {cas_number}")
                continue
            
            if i % 10 == 0:  # Aquí estaba el error: decía S0 en lugar de 0
                print(f"  Encontradas {len(tables)} tablas")
            
            # Procesar cada tabla
            for table_idx, table in enumerate(tables):
                # Extraer encabezados
                headers = []
                header_row = table.find('tr')
                if header_row:
                    headers = [th.text.strip() for th in header_row.find_all('th')]
                
                # Extraer filas
                for row_idx, tr in enumerate(table.find_all('tr')[1:]):  # Saltar la fila de encabezados
                    cells = [td.text.strip() for td in tr.find_all('td')]
                    if len(cells) != len(headers):
                        continue  # Saltar filas incompletas
                    
                    # Crear diccionario para esta fila
                    row_dict = {headers[j]: cells[j] for j in range(len(headers))}
                    
                    # Añadir información del compuesto
                    row_dict['CAS_Number'] = cas_number
                    row_dict['Compound_Name'] = clean_name
                    row_dict['NIST_Name'] = nist_name
                    row_dict['Table_Number'] = table_idx + 1
                    
                    # Guardar
                    all_data.append(row_dict)
            
            processed_count += 1
            
            # Guardar progreso intermedio cada 1000 compuestos
            if len(all_data) > 0 and processed_count % 1000 == 0:
                temp_df = pd.DataFrame(all_data)
                temp_file = f"temp_{processed_count}_retention_data.xlsx"
                temp_df.to_excel(temp_file, index=False)
                print(f"Guardado progreso temporal en {temp_file} ({len(all_data)} filas)")
        
        except Exception as e:
            print(f"  Error procesando documento {i} (CAS: {cas_number}): {e}")
            error_count += 1
            continue
    
    # Crear DataFrame
    if not all_data:
        print("No se pudieron extraer datos de las tablas.")
        return None
    
    print(f"Creando DataFrame con {len(all_data)} filas...")
    df = pd.DataFrame(all_data)
    
    # Ordenar columnas (información del compuesto primero)
    info_cols = ['Compound_Name', 'CAS_Number', 'NIST_Name', 'Table_Number']
    other_cols = [col for col in df.columns if col not in info_cols]
    
    # Reordenar columnas y asegurar que existan (por si hay datos vacíos)
    for col in info_cols:
        if col not in df.columns:
            df[col] = 'N/A'
    
    # Seleccionar solo las columnas que existen
    available_cols = [col for col in info_cols + other_cols if col in df.columns]
    df = df[available_cols]
    
    # Guardar a Excel
    print(f"Guardando datos en {output_file}...")
    df.to_excel(output_file, index=False)
    
    print(f"\nProceso completado. Datos guardados en {os.path.abspath(output_file)}")
    print(f"Resumen:")
    print(f"- Total de compuestos procesados: {processed_count}")
    print(f"- Compuestos con errores: {error_count}")
    print(f"- Total de filas extraídas: {len(df)}")
    print(f"- Total de compuestos únicos: {df['CAS_Number'].nunique()}")
    print(f"- Columnas: {', '.join(df.columns.tolist())}")
    
    return output_file

if __name__ == "__main__":
    # Ejecutar parseo con la ruta específica
    parse_retention_tables()