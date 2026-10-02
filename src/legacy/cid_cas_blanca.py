import requests
import pandas as pd
import time
import re
import os
import urllib.parse
import concurrent.futures
import shutil
from tqdm import tqdm  # Para barra de progreso

# Configuración para optimizar velocidad
MAX_WORKERS = 5  # Número de trabajadores en paralelo (ajustar según tu conexión)
SAVE_FREQUENCY = 50  # Guardar cada 50 entradas procesadas
OUTPUT_FILE = "genotoxicity_cas_cid.xlsx"
PUBCHEM_BATCH_SIZE = 20  # Procesar en lotes para evitar bloqueos de IP
MAX_RETRIES = 3  # Reintentos por solicitud

def clean_substance_name(encoded_name):
    """Limpia el nombre de la sustancia eliminando codificaciones HTML"""
    if not isinstance(encoded_name, str):
        return encoded_name
        
    replacements = {
        '_x0028_': '(', '_x0029_': ')', '_x002B_': '+', '_x002D_': '-',
        '_x0020_': ' ', '_x0027_': "'", '_x002E_': '.', '_x002C_': ',',
        '_x005B_': '[', '_x005D_': ']',
        '_x0030_': '0', '_x0031_': '1', '_x0032_': '2', '_x0033_': '3',
        '_x0034_': '4', '_x0035_': '5', '_x0036_': '6', '_x0037_': '7',
        '_x0038_': '8', '_x0039_': '9'
    }
    
    cleaned_name = encoded_name
    for code, char in replacements.items():
        cleaned_name = cleaned_name.replace(code, char)
    
    return cleaned_name

def search_cas_number_pubchem(substance_name):
    """Busca el CAS y CID en PubChem con reintentos"""
    if not substance_name or pd.isna(substance_name):
        return None, None
        
    # Codificar para URL
    encoded_name = urllib.parse.quote(substance_name)
    search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/cids/JSON"
    
    # Función para reintento con backoff exponencial
    def make_request_with_retry(url, retries=MAX_RETRIES, backoff_factor=0.5):
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:  # Rate limit
                    time.sleep((2 ** attempt) * backoff_factor)
                    continue
                else:
                    return None
            except Exception:
                time.sleep((2 ** attempt) * backoff_factor)
        return None
    
    try:
        # Buscar CID
        response = make_request_with_retry(search_url)
        if not response:
            return None, None
            
        data = response.json()
        
        # Verificar si hay resultados
        if 'IdentifierList' not in data or 'CID' not in data['IdentifierList']:
            return None, None
        
        # Obtener el primer CID
        cid = data['IdentifierList']['CID'][0]
        
        # Buscar número CAS
        synonyms_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
        syn_response = make_request_with_retry(synonyms_url)
        
        if not syn_response:
            return None, cid
        
        try:
            synonyms = syn_response.json()['InformationList']['Information'][0]['Synonym']
            
            # Buscar CAS en sinónimos
            cas_pattern = re.compile(r'^\d+-\d+-\d+$')
            cas_number = None
            
            for synonym in synonyms:
                if cas_pattern.match(synonym):
                    cas_number = synonym
                    break
                    
            return cas_number, cid
            
        except (KeyError, IndexError):
            return None, cid
        
    except Exception:
        return None, None

def process_batch(batch_data):
    """Procesa un lote de sustancias y devuelve los resultados"""
    results = []
    
    for index, row in batch_data.iterrows():
        substance = row['CleanSubstanceName']
        cas, cid = None, None
        
        # Solo buscar si los campos están vacíos
        if pd.isna(row['CAS']) or pd.isna(row['PubChem_CID']):
            cas, cid = search_cas_number_pubchem(substance)
            
            # Mantener valores existentes si no se encontraron nuevos
            if pd.isna(row['CAS']) or row['CAS'] == '':
                row['CAS'] = cas
            if pd.isna(row['PubChem_CID']) or row['PubChem_CID'] == '':
                row['PubChem_CID'] = cid
        
        results.append((index, row['CAS'], row['PubChem_CID']))
        
        # Pequeña pausa para prevenir bloqueos
        time.sleep(0.1)
        
    return results

def backup_file(filename):
    """Crea una copia de seguridad del archivo si existe"""
    if os.path.exists(filename) and os.path.getsize(filename) > 1000:
        backup = f"{filename}.bak"
        try:
            shutil.copy2(filename, backup)
            print(f"Backup creado: {backup}")
        except:
            pass

def safe_save(df, filename):
    """Guarda el DataFrame de forma segura"""
    if df.empty:
        return False
        
    try:
        # Crear backup
        backup_file(filename)
        
        # Guardar a archivo temporal
        temp_file = f"{filename}.temp.xlsx"
        df.to_excel(temp_file, index=False)
        
        # Verificar que se guardó correctamente
        if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
            # Reemplazar archivo original
            if os.path.exists(filename):
                os.remove(filename)
            os.rename(temp_file, filename)
            return True
            
    except Exception as e:
        print(f"Error al guardar: {str(e)}")
        # Intentar formato CSV como respaldo
        try:
            df.to_csv(f"{filename}.csv", index=False)
        except:
            pass
    
    return False

def main():
    print("=== Procesador Paralelo de CAS y PubChem CID ===")
    
    # Solicitar archivo de entrada
    input_file = input("Nombre del archivo Excel (default: Genotoxicity_KJ_2023.xlsx): ") or "Genotoxicity_KJ_2023.xlsx"
    
    if not os.path.exists(input_file):
        print(f"Error: Archivo {input_file} no encontrado.")
        return
    
    # Cargar archivo Excel
    try:
        print(f"Cargando {input_file}...")
        df = pd.read_excel(input_file)
        print(f"Archivo cargado con {len(df)} filas.")
    except Exception as e:
        print(f"Error al leer archivo: {e}")
        return
    
    # Verificar columna de sustancias
    if 'Substance' not in df.columns:
        substance_col = input("Columna 'Substance' no encontrada. Nombre de la columna con nombres químicos: ")
        if substance_col not in df.columns:
            print(f"Error: Columna '{substance_col}' no encontrada.")
            return
    else:
        substance_col = 'Substance'
    
    # Limpiar nombres de sustancias
    print("Limpiando nombres de sustancias...")
    df['CleanSubstanceName'] = df[substance_col].apply(clean_substance_name)
    
    # Inicializar columnas si no existen
    for col in ['CAS', 'PubChem_CID']:
        if col not in df.columns:
            df[col] = None
    
    # Solicitar índice inicial
    try:
        start_index = int(input(f"Índice inicial (0-{len(df)-1}, default: 0): ") or "0")
        if start_index < 0 or start_index >= len(df):
            print("Índice inválido. Usando 0.")
            start_index = 0
    except ValueError:
        print("Valor inválido. Iniciando desde 0.")
        start_index = 0
    
    # Solicitar número de entradas a procesar
    try:
        limit = int(input("Número de sustancias a procesar (0 para todas): ") or "0")
    except ValueError:
        print("Valor inválido. Procesando todas.")
        limit = 0
    
    # Calcular índice final
    if limit > 0:
        end_index = min(start_index + limit, len(df))
    else:
        end_index = len(df)
    
    # Obtener subconjunto de datos a procesar
    data_to_process = df.iloc[start_index:end_index].copy()
    
    # Contador de filas procesadas
    processed_count = 0
    found_cas = 0
    found_cid = 0
    
    print(f"\nProcesando {len(data_to_process)} sustancias en paralelo con {MAX_WORKERS} trabajadores")
    print(f"Guardando cada {SAVE_FREQUENCY} sustancias")
    print("Presione Ctrl+C para interrumpir\n")
    
    # Dividir en lotes para procesar en paralelo
    batch_size = PUBCHEM_BATCH_SIZE
    batches = [data_to_process.iloc[i:i+batch_size] for i in range(0, len(data_to_process), batch_size)]
    
    try:
        # Barra de progreso
        with tqdm(total=len(data_to_process)) as pbar:
            for batch_idx, batch in enumerate(batches):
                # Procesar lote en paralelo
                with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    batch_futures = []
                    for chunk_start in range(0, len(batch), MAX_WORKERS):
                        chunk = batch.iloc[chunk_start:min(chunk_start+MAX_WORKERS, len(batch))]
                        batch_futures.append(executor.submit(process_batch, chunk))
                
                    # Recopilar resultados
                    for future in concurrent.futures.as_completed(batch_futures):
                        results = future.result()
                        for index, cas, cid in results:
                            # Actualizar DataFrame principal
                            if cas and pd.isna(df.at[index, 'CAS']):
                                df.at[index, 'CAS'] = cas
                                found_cas += 1
                            
                            if cid and pd.isna(df.at[index, 'PubChem_CID']):
                                df.at[index, 'PubChem_CID'] = cid
                                found_cid += 1
                            
                            processed_count += 1
                            pbar.update(1)
                
                # Guardar periódicamente
                if (batch_idx + 1) % (SAVE_FREQUENCY // batch_size) == 0 or batch_idx == len(batches) - 1:
                    print(f"\nGuardando progreso ({processed_count}/{len(data_to_process)} completados)...")
                    safe_save(df, OUTPUT_FILE)
                    print(f"Encontrados: {found_cas} CAS, {found_cid} CID")
    
    except KeyboardInterrupt:
        print("\n\nProceso interrumpido por el usuario.")
    except Exception as e:
        print(f"\n\nError inesperado: {e}")
    finally:
        # Guardar resultados finales
        if processed_count > 0:
            print("\nGuardando resultados finales...")
            safe_save(df, OUTPUT_FILE)
            
            # Mostrar resumen
            print("\n=== RESUMEN ===")
            print(f"Procesadas: {processed_count} sustancias")
            print(f"Encontrados: {found_cas} números CAS")
            print(f"Encontrados: {found_cid} IDs de PubChem")
            print(f"Resultados guardados en: {OUTPUT_FILE}")
        else:
            print("\nNo se procesaron sustancias.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPrograma interrumpido por el usuario.")
    except Exception as e:
        print(f"\nError inesperado: {e}")
    finally:
        print("\nPrograma finalizado.")