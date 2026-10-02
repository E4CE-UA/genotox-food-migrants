import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import csv
import os

def extract_retention_data(cas_number, compound_name):
    """
    Extract gas chromatography retention data from NIST WebBook for a given CAS number.
    
    Args:
        cas_number (str): Chemical CAS registry number (e.g., "71-43-2" for benzene)
        compound_name (str): Name of the compound for reference
        
    Returns:
        list: List of dictionaries containing retention data
    """
    # Skip invalid CAS numbers
    if cas_number == "0-00-0" or not cas_number:
        print(f"  Saltando CAS inválido: {cas_number}")
        return []
    
    # Try different URL formats to find data
    urls_to_try = [
        f"https://webbook.nist.gov/cgi/cbook.cgi?ID=C{cas_number.replace('-', '')}&Units=SI&Mask=2000#Gas-Chrom",
        f"https://webbook.nist.gov/cgi/cbook.cgi?Name={compound_name.replace(' ', '+')}&Units=SI&Mask=2000#Gas-Chrom",
        f"https://webbook.nist.gov/cgi/cbook.cgi?ID={cas_number}&Units=SI&Mask=2000#Gas-Chrom"
    ]
    
    results = []
    
    for url in urls_to_try:
        try:
            # Add a small delay to avoid overwhelming the server
            time.sleep(1)
            
            # Send request to NIST WebBook
            print(f"Consultando: {url}")
            response = requests.get(url)
            
            # Skip failed requests
            if response.status_code != 200:
                print(f"  Error HTTP {response.status_code} para {url}")
                continue
                
            # Parse HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check if we have data or a "no data available" message
            no_data_msg = soup.find(string=re.compile("No gas chromatography data found"))
            if no_data_msg:
                print(f"  No hay datos de cromatografía de gases para este compuesto según NIST")
                continue
                
            # Extract compound name from NIST (might be more accurate than the database name)
            nist_compound_name = None
            h1_tag = soup.find('h1')
            if h1_tag:
                nist_compound_name = h1_tag.text.strip()
            
            # Find all data tables
            data_tables = soup.find_all('table', {'class': 'data'})
            
            # Debug information
            print(f"  Nombre en NIST: {nist_compound_name if nist_compound_name else 'No encontrado'}")
            print(f"  Tablas de datos encontradas: {len(data_tables)}")
            
            # Look through all tables for retention data
            found_retention_data = False
            for table in data_tables:
                # Get table headers
                headers = []
                th_tags = table.find_all('th')
                for th in th_tags:
                    headers.append(th.text.strip())
                
                # Check if this table has retention data (look for relevant keywords)
                retention_keywords = ['retention', 'tiempo', 'índice', 'column', 'columna']
                is_retention_table = any(keyword in ' '.join(headers).lower() for keyword in retention_keywords)
                
                if is_retention_table:
                    found_retention_data = True
                    print(f"  Encontrada tabla de retención con encabezados: {headers}")
                    
                    # Process rows
                    for row in table.find_all('tr')[1:]:  # Skip header row
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            # Create a dictionary for each data point
                            # Mapping depends on the table structure, which can vary
                            retention_data = {
                                'CAS': cas_number,
                                'NISTCompoundName': nist_compound_name
                            }
                            
                            # Map cells to data fields based on headers
                            for i, header in enumerate(headers):
                                if i < len(cells):
                                    header_lower = header.lower()
                                    cell_value = cells[i].text.strip()
                                    
                                    # Map common headers to standardized fields
                                    if 'column' in header_lower or 'columna' in header_lower:
                                        retention_data['Column'] = cell_value
                                    elif 'index' in header_lower or 'índice' in header_lower or 'indice' in header_lower:
                                        retention_data['RetentionIndex'] = cell_value
                                    elif 'temp' in header_lower:
                                        retention_data['Temperature'] = cell_value
                                    elif 'time' in header_lower or 'tiempo' in header_lower:
                                        retention_data['RetentionTime'] = cell_value
                                    elif 'ref' in header_lower:
                                        retention_data['Reference'] = cell_value
                                    else:
                                        # For any unmapped header, store with original name
                                        retention_data[header] = cell_value
                            
                            results.append(retention_data)
            
            if not found_retention_data:
                print(f"  No se encontraron tablas específicas de retención cromatográfica")
                
                # Try a different approach: look for specific sections
                # Sometimes data is in sections rather than tables
                gc_section = soup.find(id="Gas-Chrom")
                if gc_section:
                    print(f"  Encontrada sección de cromatografía de gases, extrayendo datos...")
                    # Extract data from the section (implementation depends on structure)
                    # This is a simplified approach; may need customization based on actual HTML structure
                    retention_data = {
                        'CAS': cas_number,
                        'NISTCompoundName': nist_compound_name,
                        'DataExtracted': 'Extracted from Gas-Chrom section, format varies'
                    }
                    results.append(retention_data)
            
            # If we found data, no need to try other URLs
            if results:
                break
                
        except requests.exceptions.RequestException as e:
            print(f"  Error de conexión: {e}")
        except Exception as e:
            print(f"  Error al procesar datos: {e}")
    
    if not results:
        print(f"  No se encontraron datos de retención tras intentar múltiples URLs")
    else:
        print(f"  Se encontraron {len(results)} puntos de datos de retención")
        
    return results

def read_csv_file(file_path_or_content):
    """
    Read compounds from CSV file or content string.
    
    Args:
        file_path_or_content (str): Path to CSV file or CSV content string
        
    Returns:
        list: List of dictionaries containing compound information
    """
    compounds = []
    
    # Check if input is a file path or content string
    is_file_path = os.path.isfile(file_path_or_content)
    
    try:
        if is_file_path:
            # Read from file
            df = pd.read_csv(file_path_or_content)
        else:
            # Read from string content
            lines = file_path_or_content.strip().split('\n')
            # If there's a header row
            if lines and ',' in lines[0]:
                header = lines[0].split(',')
                data_rows = []
                
                # Process each data row
                for line in lines[1:]:
                    if not line.strip():
                        continue
                        
                    # Parse CSV line handling quoted values
                    row_values = []
                    in_quotes = False
                    current_value = ""
                    
                    for char in line:
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            row_values.append(current_value)
                            current_value = ""
                        else:
                            current_value += char
                    
                    # Add the last value
                    row_values.append(current_value)
                    
                    # Create row dictionary
                    row_dict = {}
                    for i, col_name in enumerate(header):
                        if i < len(row_values):
                            row_dict[col_name.strip()] = row_values[i].strip().strip('"')
                    
                    data_rows.append(row_dict)
                
                df = pd.DataFrame(data_rows)
            else:
                # No valid data
                return []
        
        # Extract compounds from DataFrame
        for _, row in df.iterrows():
            compound = {}
            
            # Map DataFrame columns to compound fields
            for col in df.columns:
                # Clean up column names
                clean_col = col.strip()
                value = str(row[col]).strip()
                
                # Handle specific column names
                if clean_col.lower() == 'cas_' or clean_col.lower() == 'cas':
                    compound['CAS'] = value
                elif clean_col.lower() == 'compoundname':
                    compound['CompoundName'] = value.strip('"')
                else:
                    compound[clean_col] = value
            
            # Ensure key fields exist
            if 'CAS' not in compound and 'CAS_' in compound:
                compound['CAS'] = compound['CAS_']
            if 'CompoundName' not in compound:
                # Try to find a column that might contain compound names
                for col in compound:
                    if 'name' in col.lower() or 'compound' in col.lower():
                        compound['CompoundName'] = compound[col]
                        break
            
            compounds.append(compound)
    
    except Exception as e:
        print(f"Error al leer datos CSV: {e}")
        
        # Fall back to manual parsing for the specific format
        if not is_file_path:
            try:
                lines = file_path_or_content.strip().split('\n')
                for line in lines:
                    if not line.strip() or line.startswith("SampleName"):
                        continue
                    
                    # Try to parse each line based on your specific format
                    parts = line.split(',')
                    
                    # Extract sample name
                    sample_name = parts[0]
                    
                    # Extract compound name (which is in quotes)
                    compound_start = line.find('"')
                    compound_end = line.find('"', compound_start + 1)
                    
                    if compound_start >= 0 and compound_end > compound_start:
                        compound_name = line[compound_start + 1:compound_end]
                        remaining = line[compound_end + 2:].split(',')
                        
                        if len(remaining) >= 5:
                            compounds.append({
                                'SampleName': sample_name,
                                'CompoundName': compound_name,
                                'ComponentRT': remaining[0],
                                'MatchFactor': remaining[1],
                                'Formula': remaining[2],
                                'CAS': remaining[3],
                                'ComponentArea': remaining[4],
                                'ModelPeakMZ': remaining[5] if len(remaining) > 5 else ''
                            })
            except Exception as e:
                print(f"Error en parseo manual: {e}")
    
    return compounds

def main():
    print("=== NIST WebBook - Extractor de datos de retención cromatográfica ===")
    print("Este script procesa compuestos y consulta NIST WebBook para obtener")
    print("sus datos de retención cromatográfica.")
    print("------------------------------------------------------------------")
    
    # Determinar fuente de datos
    print("\nOpciones de fuente de datos:")
    print("1. Usar archivo CSV (AF016.csv)")
    print("2. Usar datos de ejemplo predefinidos")
    print("3. Ingresar CAS específico manualmente")
    
    try:
        data_source = int(input("\nSeleccione una opción (1-3): "))
    except ValueError:
        print("Opción inválida. Usando archivo CSV por defecto.")
        data_source = 1
    
    compounds = []
    
    if data_source == 1:
        # Usar archivo CSV
        filename = input("Nombre del archivo CSV (Enter para usar 'AF016.csv'): ") or "AF016.csv"
        try:
            print(f"Intentando leer el archivo {filename}...")
            compounds = read_csv_file(filename)
            if not compounds:
                raise FileNotFoundError(f"No se pudo leer {filename} o está vacío")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Cambiando a datos de ejemplo...")
            data_source = 2
    
    if data_source == 2:
        # Usar datos de ejemplo
        example_data = """SampleName,CompoundName,ComponentRT,MatchFactor,Formula,CAS_,ComponentArea,ModelPeakMZ
AF016,"Methane, difluoro-",8.14764361636854,72.3184638023138,CH2F2,75-10-5,100109.54185529,48.5290050496024
AF016,"Benzene",8.18400874540069,53.6624648730282,C6H6,71-43-2,1482.94974951937,49.8693351689579
AF016,"Toluene",8.19672795147963,79.6137271542164,C7H8,108-88-3,167717.3760306,50.5191891876128"""
        
        print("Usando datos de ejemplo predefinidos.")
        compounds = read_csv_file(example_data)
    
    if data_source == 3:
        # Ingresar CAS manualmente
        cas = input("Ingrese el número CAS (ej. 71-43-2 para benceno): ")
        name = input("Ingrese el nombre del compuesto (ej. Benzene): ")
        compounds = [{
            'SampleName': 'Manual',
            'CompoundName': name,
            'CAS': cas,
            'Formula': input("Ingrese la fórmula (opcional): ") or "N/A"
        }]
    
    # Obtener el límite de componentes a procesar
    try:
        limit = int(input("\nIntroduce el número de componentes a procesar (0 para todos): "))
    except ValueError:
        print("Valor no válido. Se procesarán todos los componentes.")
        limit = 0
    
    # Aplicar el límite
    if limit > 0 and limit < len(compounds):
        compounds = compounds[:limit]
        print(f"Se procesarán {len(compounds)} compuestos (limitado por el usuario).")
    else:
        print(f"Se procesarán {len(compounds)} compuestos.")
    
    # Inicializar contadores
    cas_total = len(compounds)
    cas_con_datos = 0
    cas_sin_datos = 0
    puntos_datos_total = 0
    
    # Procesar cada compuesto
    all_results = []
    
    for i, compound in enumerate(compounds):
        cas = compound.get('CAS', '')
        compound_name = compound.get('CompoundName', '')
        
        print(f"\n[{i+1}/{len(compounds)}] Procesando {compound_name} (CAS: {cas})...")
        
        # Skip invalid CAS numbers
        if cas == "0-00-0" or not cas:
            print(f"  Saltando CAS inválido: {cas}")
            cas_sin_datos += 1
            continue
        
        # Extract retention data from NIST WebBook
        retention_data = extract_retention_data(cas, compound_name)
        
        # Actualizar contadores
        if retention_data:
            cas_con_datos += 1
            puntos_datos_total += len(retention_data)
        else:
            cas_sin_datos += 1
        
        # Add original compound info to each retention data point
        for data in retention_data:
            # Add all original compound data fields
            for key, value in compound.items():
                if key not in data:  # Don't overwrite existing fields
                    data['Original_' + key] = value
            
            all_results.append(data)
    
    # Mostrar resumen de resultados
    print("\n=== RESUMEN DE RESULTADOS ===")
    print(f"Total de CAS procesados: {cas_total}")
    print(f"CAS con datos de retención: {cas_con_datos} ({cas_con_datos/cas_total*100:.1f}%)")
    print(f"CAS sin datos de retención: {cas_sin_datos} ({cas_sin_datos/cas_total*100:.1f}%)")
    print(f"Total de puntos de datos obtenidos: {puntos_datos_total}")
    print(f"Promedio de puntos por CAS con datos: {puntos_datos_total/cas_con_datos if cas_con_datos else 0:.1f}")
    
    # Save results to CSV
    if all_results:
        # Generate output filename
        output_filename = "nist_retention_data.csv"
        if data_source == 1:
            # Use input filename as base for output filename
            base_name = os.path.splitext(os.path.basename(filename))[0]
            output_filename = f"{base_name}_nist_retention_data.csv"
        
        # Get all possible field names from all results
        all_fields = set()
        for result in all_results:
            all_fields.update(result.keys())
        
        # Prioritize certain fields to appear first
        priority_fields = [
            'CAS', 'NISTCompoundName', 'Original_CompoundName', 'Original_SampleName',
            'RetentionIndex', 'RetentionTime', 'Column', 'Temperature', 'Reference'
        ]
        
        # Sort fields: priority fields first, then remaining fields alphabetically
        fields = [f for f in priority_fields if f in all_fields]
        remaining_fields = sorted(list(all_fields - set(fields)))
        fields.extend(remaining_fields)
        
        # Write to CSV
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\nGuardados {len(all_results)} puntos de datos de retención en {output_filename}")
    else:
        print("\nNo se encontraron datos de retención.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPrograma interrumpido por el usuario.")
    except Exception as e:
        print(f"\n\nError inesperado: {e}")
    finally:
        print("\nPrograma finalizado.")