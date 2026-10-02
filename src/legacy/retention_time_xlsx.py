import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import openpyxl
from openpyxl.styles import Font
import chromadb
from chromadb.utils import embedding_functions
import hashlib
from datetime import datetime

# Global result queue for thread-safe data collection
result_queue = queue.Queue()

# Configurar ChromaDB con persistencia
# Crear directorio para la base de datos si no existe
os.makedirs("./chroma_db", exist_ok=True)
client = chromadb.PersistentClient(path="./chroma_db")
try:
    collection = client.get_collection("nist_retention_data")
    print("Colección ChromaDB existente encontrada")
except:
    collection = client.create_collection(
        name="nist_retention_data",
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )
    print("Nueva colección ChromaDB creada")

def generate_document_id(cas_number):
    """Genera un ID único para un documento basado en el número CAS"""
    return hashlib.md5(cas_number.encode()).hexdigest()

def extract_retention_tables(html_content):
    """
    Extract tables containing retention data from HTML content
    
    Args:
        html_content (str): HTML content from NIST WebBook
        
    Returns:
        list: List of tables HTML as strings
    """
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

def extract_basic_retention_data(cas_number, substance_name, row_index):
    """
    Extract retention data from NIST WebBook for a given CAS number.
    Store HTML content in ChromaDB for later analysis.
    
    Args:
        cas_number (str): Chemical CAS registry number
        substance_name (str): Name of the substance for reference
        row_index (int): Index of the row in the dataframe
        
    Returns:
        dict: Dictionary with extracted data and row index
    """
    if not cas_number or str(cas_number).lower() == 'na' or str(cas_number).strip() == "":
        print(f"  No valid CAS number for {substance_name}, skipping...")
        return {'row_index': row_index, 'CAS': cas_number}
    
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
        'row_index': row_index,
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
        'doc_id': None
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
    
    return results

def worker_function(cas_number, substance_name, row_index, max_retries=3):
    """
    Worker function for thread pool to handle retries
    
    Args:
        cas_number (str): CAS number
        substance_name (str): Substance name
        row_index (int): DataFrame row index
        max_retries (int): Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            # Extract basic data and store in ChromaDB
            result = extract_basic_retention_data(cas_number, substance_name, row_index)
            
            # Clean up temporary fields
            if 'html_stored' in result:
                del result['html_stored']
            
            # Put result in the queue
            result_queue.put(result)
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt+1} for {cas_number}: {e}")
                time.sleep(2)  # Wait before retry
            else:
                print(f"Failed after {max_retries} attempts for {cas_number}: {e}")
                result_queue.put({
                    'row_index': row_index, 
                    'CAS': cas_number, 
                    'Error': f"Failed after {max_retries} attempts: {e}"
                })

def process_compounds_in_parallel(df, start_index, end_index, max_workers=8):
    """
    Process compounds in parallel using thread pool
    
    Args:
        df (DataFrame): Pandas DataFrame with compounds
        start_index (int): Starting index
        end_index (int): Ending index
        max_workers (int): Maximum number of worker threads
        
    Returns:
        dict: Results dictionary with row indices as keys
    """
    print(f"Processing {end_index - start_index} compounds with {max_workers} parallel workers")
    
    # Configure thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit tasks
        for i in range(start_index, end_index):
            row = df.iloc[i]
            cas = row['CAS'] if not pd.isna(row['CAS']) else None
            
            # Try to get substance name from multiple possible column names
            substance = None
            for col_name in ['Substance', 'CompoundName', 'Name', 'Compound']:
                if col_name in df.columns and not pd.isna(row[col_name]):
                    substance = row[col_name]
                    break
            
            if not substance:
                substance = f"Compound_{i}"
            
            # Submit task to thread pool
            executor.submit(worker_function, cas, substance, i)
    
    # Collect results from queue
    results = {}
    while not result_queue.empty():
        result = result_queue.get()
        row_index = result.pop('row_index')
        results[row_index] = result
    
    return results

def update_dataframe_with_results(df, results):
    """
    Update dataframe with results from parallel processing
    
    Args:
        df (DataFrame): Pandas DataFrame
        results (dict): Results dictionary with row indices as keys
        
    Returns:
        DataFrame: Updated DataFrame
    """
    # Add columns if needed
    columns_to_add = [
        'NISTName', 'NonPolarRI', 'NonPolarRI_Temp', 'NonPolarRI_Column', 'NonPolarRI_Source',
        'PolarRI', 'PolarRI_Temp', 'PolarRI_Column', 'PolarRI_Source',
        'doc_id', 'Error'
    ]
    
    for col in columns_to_add:
        if col not in df.columns:
            df[col] = None
    
    # Update with results
    for row_index, result in results.items():
        for key, value in result.items():
            if key in df.columns and value is not None:
                df.at[row_index, key] = value
    
    return df

def process_batch_parallel(input_file, batch_size=50, max_workers=8, output_file="retention_times_chroma.xlsx"):
    """
    Process a batch of CAS numbers from an Excel file in parallel and save results
    
    Args:
        input_file (str): Excel file containing CAS numbers
        batch_size (int): Number of CAS numbers to process in this batch
        max_workers (int): Maximum number of parallel workers
        output_file (str): Output file name
    
    Returns:
        str: Path to output file
    """
    # Read Excel file
    print(f"Reading {input_file}...")
    df = pd.read_excel(input_file)
    print(f"Found {len(df)} entries")
    
    # Check if 'CAS' column exists
    if 'CAS' not in df.columns:
        print("Error: No 'CAS' column found in the input file")
        return None
    
    # Check for save point
    save_point_file = f"{os.path.splitext(input_file)[0]}_save_point.txt"
    start_index = 0
    
    if os.path.exists(save_point_file):
        with open(save_point_file, 'r') as f:
            saved_index = int(f.read().strip())
            
        print(f"Found save point at index {saved_index}")
        resume = input(f"Resume from index {saved_index}? (y/n, default: y): ") or "y"
        
        if resume.lower() == "y":
            start_index = saved_index
            print(f"Resuming from index {start_index}")
            # Check if output file exists
            if os.path.exists(output_file):
                print(f"Loading previous results from {output_file}")
                df = pd.read_excel(output_file)
        else:
            try:
                start_index = int(input(f"Enter starting index (0-{len(df)-1}, default: 0): ") or "0")
                if start_index < 0 or start_index >= len(df):
                    print(f"Invalid index. Using 0.")
                    start_index = 0
            except ValueError:
                print("Invalid value. Starting from the beginning.")
                start_index = 0
    else:
        try:
            start_index = int(input(f"Enter starting index (0-{len(df)-1}, default: 0): ") or "0")
            if start_index < 0 or start_index >= len(df):
                print(f"Invalid index. Using 0.")
                start_index = 0
        except ValueError:
            print("Invalid value. Starting from the beginning.")
            start_index = 0
    
    # Calculate end index
    end_index = min(start_index + batch_size, len(df))
    print(f"Processing entries {start_index} to {end_index-1} (total: {end_index-start_index})")
    
    # Process in parallel
    start_time = time.time()
    results = process_compounds_in_parallel(df, start_index, end_index, max_workers)
    end_time = time.time()
    
    # Update dataframe with results
    df = update_dataframe_with_results(df, results)
    
    # Save results
    print(f"\nProcessed {len(results)} compounds in {end_time - start_time:.2f} seconds")
    print(f"Saving results to {output_file}")
    df.to_excel(output_file, index=False)
    
    # Create a save point file
    with open(save_point_file, 'w') as f:
        f.write(str(end_index))
    print(f"Save point created at index {end_index} in {save_point_file}")
    
    # Calculate statistics
    found_data_count = sum(1 for result in results.values() if (
        result.get('NonPolarRI') or result.get('PolarRI') or result.get('doc_id')
    ))
    print(f"Found retention data for {found_data_count} out of {len(results)} compounds ({found_data_count/len(results)*100:.1f}%)")
    
    return output_file

def main():
    print("=== NIST Retention Data Extractor with ChromaDB Persistente ===")
    print("Este script extrae tablas de retención de NIST WebBook")
    print("y las almacena en ChromaDB (con persistencia) para análisis posterior.")
    print("Ubicación de la base de datos: ./chroma_db")
    print("------------------------------------------------------------------")
    
    # Get input file
    input_file = input("Enter the Excel file name (default: genotoxicity_cas_cid.xlsx): ") or "genotoxicity_cas_cid.xlsx"
    
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return
    
    # Get batch size
    try:
        batch_size = int(input("How many compounds to process in this batch? (default: 50): ") or "50")
        if batch_size <= 0:
            print("Invalid batch size. Using 50.")
            batch_size = 50
    except ValueError:
        print("Invalid value. Using default batch size of 50.")
        batch_size = 50
    
    # Get number of worker threads
    try:
        max_workers = int(input("How many parallel workers to use? (default: 8): ") or "8")
        if max_workers <= 0:
            print("Invalid number of workers. Using 8.")
            max_workers = 8
        elif max_workers > 20:
            print("Warning: Using too many workers may cause the server to block your requests.")
            confirm = input("Are you sure you want to use more than 20 workers? (y/n, default: n): ") or "n"
            if confirm.lower() != "y":
                max_workers = 20
                print("Using 20 workers instead.")
    except ValueError:
        print("Invalid value. Using default of 8 workers.")
        max_workers = 8
    
    # Get output file name
    output_file = input("Enter output file name (default: retention_times_chroma.xlsx): ") or "retention_times_chroma.xlsx"
    
    # Process batch
    output_file = process_batch_parallel(input_file, batch_size, max_workers, output_file)
    
    if output_file:
        print(f"\nBatch processing complete. Results saved to {output_file}")
        print(f"Los datos de ChromaDB se han guardado en ./chroma_db")
        
        # Ask if user wants to process another batch
        next_batch = input("Process another batch? (y/n, default: n): ") or "n"
        if next_batch.lower() == "y":
            main()
    else:
        print("\nBatch processing failed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
    finally:
        print("\nProgram finished.")