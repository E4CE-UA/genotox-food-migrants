import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures  # Añadir esta línea
import threading
import queue
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from datetime import datetime
from tqdm import tqdm  # Para barra de progreso

# Global result queue for thread-safe data collection
result_queue = queue.Queue()

# Configurar ChromaDB con persistencia
# Crear directorio para la base de datos si no existe
os.makedirs("./chroma_db", exist_ok=True)
client = chromadb.PersistentClient(path="./chroma_db")

# Nombre de la nueva colección
NEW_COLLECTION_NAME = "nist_retention_data_unique"

# Crear nueva colección
try:
    # Intentar eliminar la colección existente si existe
    try:
        client.delete_collection(NEW_COLLECTION_NAME)
        print(f"Colección anterior '{NEW_COLLECTION_NAME}' eliminada")
    except:
        pass
    
    # Crear nueva colección
    collection = client.create_collection(
        name=NEW_COLLECTION_NAME,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    print(f"Nueva colección '{NEW_COLLECTION_NAME}' creada")
except Exception as e:
    print(f"Error al crear colección: {e}")
    exit(1)

def generate_document_id(cas_number):
    """Genera un ID único para un documento basado en el número CAS"""
    return hashlib.md5(cas_number.encode()).hexdigest()

def extract_retention_tables(html_content):
    """Extract tables containing retention data from HTML content"""
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = []
    
    # Find all data tables
    data_tables = soup.find_all('table', {'class': 'data'})
    
    # Process each table to find retention data
    for table in data_tables:
        # Get table headers
        headers = []
        th_tags = table.find_all('th')
        for th in th_tags:
            headers.append(th.text.strip())
        
        # Check if this table has retention data (look for relevant keywords)
        retention_keywords = ['retention', 'index', 'kovats', 'van den dool', 'tiempo', 'índice', 'column', 'columna']
        is_retention_table = any(keyword in ' '.join(headers).lower() for keyword in retention_keywords)
        
        if is_retention_table:
            # Convert table to a simplified HTML format
            simplified_table = "<table>\n"
            
            # Add headers
            simplified_table += "<tr>\n"
            for header in headers:
                simplified_table += f"<th>{header}</th>\n"
            simplified_table += "</tr>\n"
            
            # Add rows
            for row in table.find_all('tr')[1:]:  # Skip header row
                simplified_table += "<tr>\n"
                for cell in row.find_all('td'):
                    simplified_table += f"<td>{cell.text.strip()}</td>\n"
                simplified_table += "</tr>\n"
                
            simplified_table += "</table>\n"
            tables.append(simplified_table)
    
    return tables

def extract_basic_retention_data(cas_number, substance_name):
    """Extract retention data from NIST WebBook for a given CAS number."""
    if not cas_number or str(cas_number).lower() == 'na' or str(cas_number).strip() == "":
        print(f"  No valid CAS number for {substance_name}, skipping...")
        return None
    
    # Clean CAS number format
    cas_number = str(cas_number).strip()
    
    # Try different URL formats to find data
    urls_to_try = [
        f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas_number.replace('-', '')}&Units=SI&Mask=2000#Gas-Chrom",
        f"https://webbook.nist.gov/cgi/cbook.cgi?Name={substance_name.replace(' ', '+')}&Units=SI&Mask=2000#Gas-Chrom",
        f"https://webbook.nist.gov/cgi/cbook.cgi?ID={cas_number}&Units=SI&Mask=2000#Gas-Chrom"
    ]
    
    # Initialize results dictionary
    results = {
        'CAS': cas_number,
        'NISTName': None,
        'NonPolarRI': None, 
        'NonPolarRI_Temp': None,
        'NonPolarRI_Column': None,
        'NonPolarRI_Source': None,
        'PolarRI': None,
        'PolarRI_Temp': None,
        'PolarRI_Column': None,
        'PolarRI_Source': None,
        'html_stored': False,
        'doc_id': None,
        'retention_tables_count': 0
    }
    
    html_content = None
    retention_tables = []
    
    for url in urls_to_try:
        try:
            print(f"Consultando: {url}")
            
            # Add delay to prevent overwhelming the server
            time.sleep(1)
            
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                print(f"  Error HTTP {response.status_code} para {cas_number}")
                continue
            
            # Get HTML content
            html_content = response.text
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract compound name from NIST
            nist_name = None
            h1_tag = soup.find('h1')
            if h1_tag:
                nist_name = h1_tag.text.strip()
                results['NISTName'] = nist_name
                print(f"  NIST Name: {nist_name}")
            
            # Check if gas chromatography data exists
            no_data_msg = soup.find(string=re.compile("No gas chromatography data found"))
            if no_data_msg:
                print(f"  No gas chromatography data found for {cas_number}")
                continue
            
            # Find gas chromatography section
            gc_section = soup.find(id="Gas-Chrom")
            if not gc_section:
                print(f"  No Gas Chromatography section found for {cas_number}")
                continue
            
            # Extract retention tables
            retention_tables = extract_retention_tables(html_content)
            
            # Find all data tables
            data_tables = soup.find_all('table', {'class': 'data'})
            print(f"  Found {len(data_tables)} data tables for {cas_number}, {len(retention_tables)} retention tables")
            
            # Get all section headers (h3 tags)
            section_headers = soup.find_all('h3')
            
            data_found = False
            
            # Process each table for basic data extraction
            for i, table in enumerate(data_tables):
                # Get table headers
                headers = [th.text.strip() for th in table.find_all('th')]
                header_text = ' '.join(headers).lower()
                
                # Skip tables without retention information
                retention_keywords = ['retention', 'index', 'kovats', 'van den dool', 'tiempo', 'índice', 'column', 'columna']
                if not any(keyword in header_text for keyword in retention_keywords):
                    continue
                
                # Try to get the section title (h3) for this table
                section_title = ""
                if i < len(section_headers):
                    section_title = section_headers[i].text.strip().lower()
                
                # Determine if this is polar or non-polar data
                is_polar = 'polar' in section_title
                table_type = 'Polar' if is_polar else 'NonPolar'
                
                print(f"  Processing {table_type} table: {section_title}")
                
                # Get column indices for relevant data
                col_idx = {}
                for idx, header in enumerate(headers):
                    header_lower = header.lower()
                    if 'phase' in header_lower or 'active' in header_lower:
                        col_idx['column'] = idx
                    elif 'temp' in header_lower:
                        col_idx['temp'] = idx
                    elif header_lower == 'i' or 'index' in header_lower:
                        col_idx['index'] = idx
                    elif 'reference' in header_lower or 'ref' in header_lower:
                        col_idx['source'] = idx
                
                # Process rows - just take the first valid row for simplicity
                for row in table.find_all('tr')[1:5]:  # Just check first few rows
                    cells = [td.text.strip() for td in row.find_all('td')]
                    if len(cells) < 2:  # Need at least a few cells to be valid
                        continue
                    
                    if table_type == 'NonPolar' and not results['NonPolarRI']:
                        if 'index' in col_idx and col_idx['index'] < len(cells):
                            results['NonPolarRI'] = cells[col_idx['index']]
                            data_found = True
                        if 'temp' in col_idx and col_idx['temp'] < len(cells):
                            results['NonPolarRI_Temp'] = cells[col_idx['temp']]
                        if 'column' in col_idx and col_idx['column'] < len(cells):
                            results['NonPolarRI_Column'] = cells[col_idx['column']]
                        if 'source' in col_idx and col_idx['source'] < len(cells):
                            source_text = cells[col_idx['source']]
                            # Clean up the source text (remove HTML)
                            results['NonPolarRI_Source'] = re.sub(r'<[^>]+>', '', source_text)
                        break
                    
                    elif table_type == 'Polar' and not results['PolarRI']:
                        if 'index' in col_idx and col_idx['index'] < len(cells):
                            results['PolarRI'] = cells[col_idx['index']]
                            data_found = True
                        if 'temp' in col_idx and col_idx['temp'] < len(cells):
                            results['PolarRI_Temp'] = cells[col_idx['temp']]
                        if 'column' in col_idx and col_idx['column'] < len(cells):
                            results['PolarRI_Column'] = cells[col_idx['column']]
                        if 'source' in col_idx and col_idx['source'] < len(cells):
                            source_text = cells[col_idx['source']]
                            # Clean up the source text (remove HTML)
                            results['PolarRI_Source'] = re.sub(r'<[^>]+>', '', source_text)
                        break
            
            # If we found data or retention tables, we can stop searching
            if data_found or retention_tables:
                print(f"  Found retention data for {cas_number}")
                break
            else:
                print(f"  No specific retention index data found for {cas_number} in this URL")
                
        except requests.exceptions.RequestException as e:
            print(f"  Connection error for {cas_number}: {e}")
            continue
        except Exception as e:
            print(f"  Error processing data for {cas_number}: {e}")
            continue
    
    # Store HTML in ChromaDB if we found any retention tables
    if html_content and retention_tables:
        try:
            # Generate document ID
            doc_id = generate_document_id(cas_number)
            results['doc_id'] = doc_id
            results['retention_tables_count'] = len(retention_tables)
            
            # Create metadata
            metadata = {
                "cas": cas_number,
                "substance_name": substance_name,
                "nist_name": results['NISTName'] if results['NISTName'] else substance_name,
                "has_retention_data": bool(results['NonPolarRI'] or results['PolarRI'] or retention_tables),
                "tables_count": len(retention_tables),
                "timestamp": datetime.now().isoformat()
            }
            
            # Format tables for storage
            tables_text = ""
            for i, table in enumerate(retention_tables):
                tables_text += f"Table {i+1}:\n{table}\n\n"
            
            # Store in ChromaDB - just store the tables, not the whole HTML
            collection.upsert(
                ids=[doc_id],
                documents=[tables_text],
                metadatas=[metadata]
            )
            
            results['html_stored'] = True
            print(f"  Retention tables stored in ChromaDB with ID: {doc_id}")
        except Exception as e:
            print(f"  Error storing in ChromaDB: {e}")
    
    # Check if we found any data after trying all URLs
    if not results['NonPolarRI'] and not results['PolarRI'] and not results['html_stored']:
        print(f"  No retention data found for {cas_number} after trying all URLs")
        return None
    
    return results

def process_unique_cas_numbers(input_file, max_workers=10, batch_size=100):
    """Process only unique CAS numbers from the input file"""
    
    # Load the input file
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    print(f"Found {len(df)} entries")
    
    # Check if 'CAS' column exists
    if 'CAS' not in df.columns:
        print("Error: No 'CAS' column found in the input file")
        return None
    
    # Get unique CAS numbers
    unique_cas = df['CAS'].dropna().unique()
    print(f"Found {len(unique_cas)} unique CAS numbers")
    
    # Ask for confirmation
    confirm = input(f"¿Procesar {len(unique_cas)} números CAS únicos? Esto puede tardar horas. (s/n): ")
    if confirm.lower() != 's':
        print("Operación cancelada por el usuario.")
        return
    
    # Get substance names for CAS numbers
    cas_to_substance = {}
    for cas in unique_cas:
        # Get substance name from the first matching row
        matching_rows = df[df['CAS'] == cas]
        if not matching_rows.empty:
            # Try to get name from different possible columns
            for col_name in ['Substance', 'CompoundName', 'Name', 'Compound']:
                if col_name in matching_rows.columns and not pd.isna(matching_rows.iloc[0][col_name]):
                    cas_to_substance[cas] = matching_rows.iloc[0][col_name]
                    break
            
            # If no name found in columns, use default
            if cas not in cas_to_substance:
                cas_to_substance[cas] = f"Compound_{cas}"
    
    # Process in batches
    all_results = []
    total_with_retention = 0
    
    # Process in batches with progress bar
    for i in range(0, len(unique_cas), batch_size):
        batch = unique_cas[i:i+batch_size]
        print(f"\nProcesando lote {i//batch_size + 1}/{(len(unique_cas)-1)//batch_size + 1} ({i}-{min(i+batch_size-1, len(unique_cas)-1)})")
        
        # Process batch in parallel
        batch_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_basic_retention_data, cas, cas_to_substance.get(cas, f"Compound_{cas}")): cas for cas in batch}
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Procesando"):
                cas = futures[future]
                try:
                    result = future.result()
                    if result:
                        batch_results.append(result)
                except Exception as e:
                    print(f"Error procesando {cas}: {e}")
        
        # Count results with retention data
        with_retention = sum(1 for result in batch_results if result.get('retention_tables_count', 0) > 0)
        total_with_retention += with_retention
        
        print(f"Lote completado: {len(batch_results)}/{len(batch)} compuestos con datos, {with_retention} con tablas de retención")
        all_results.extend(batch_results)
        
        # Check for ChromaDB documents
        count_chroma = len(collection.get()['ids'])
        print(f"Documentos en ChromaDB: {count_chroma}")
        
        # Save intermediate results
        df_results = pd.DataFrame(all_results)
        if not df_results.empty:
            df_results.to_excel(f"unique_cas_results_{i+len(batch)}.xlsx", index=False)
            print(f"Resultados intermedios guardados en unique_cas_results_{i+len(batch)}.xlsx")
    
    # Final results
    print("\n=== RESULTADOS FINALES ===")
    print(f"Total de CAS únicos procesados: {len(unique_cas)}")
    print(f"Compuestos con datos de retención encontrados: {total_with_retention}")
    
    # Get final count from ChromaDB
    final_count = len(collection.get()['ids'])
    print(f"Documentos guardados en ChromaDB: {final_count}")
    
    # Save all results
    df_all = pd.DataFrame(all_results)
    if not df_all.empty:
        output_file = "all_unique_cas_results.xlsx"
        df_all.to_excel(output_file, index=False)
        print(f"Todos los resultados guardados en {output_file}")
    
    return final_count

if __name__ == "__main__":
    # Get input file
    input_file = input("Archivo Excel con CAS/CID (default: genotoxicity_cas_cid.xlsx): ") or "genotoxicity_cas_cid.xlsx"
    
    if not os.path.exists(input_file):
        print(f"Error: Archivo {input_file} no encontrado.")
        exit(1)
    
    # Get max workers
    try:
        max_workers = int(input("Número máximo de trabajadores en paralelo (default: 10): ") or "10")
    except ValueError:
        print("Valor inválido. Usando 10 trabajadores.")
        max_workers = 10
    
    # Get batch size
    try:
        batch_size = int(input("Tamaño de lote (default: 100): ") or "100")
    except ValueError:
        print("Valor inválido. Usando lotes de 100.")
        batch_size = 100
    
    # Process unique CAS numbers
    process_unique_cas_numbers(input_file, max_workers, batch_size)