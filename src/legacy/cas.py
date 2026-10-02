import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
import chromadb
from openai import OpenAI
from tqdm import tqdm
import re

# ==================== CONFIGURACIÓN DEL LOGGER ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chemical_safety_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURACIÓN ====================
# Nota: Es recomendable no exponer la API key en el código y utilizar variables de entorno.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_CHUNK_SIZE = 4000  # ~1000 tokens
CHROMADB_COLLECTION = "chemicals_collection"
CHROMA_PERSIST_DIR = "./chroma_db"  # Directorio para persistencia
OUTPUT_DIR = "chemicals_database"
TOXICITY_DIR = os.path.join(OUTPUT_DIR, "toxicity_analysis")
SAFETY_DIR = os.path.join(OUTPUT_DIR, "safety_analysis")
API_SLEEP_TIME = 1
GPT_MODEL = "gpt-4o"

# Inicializar cliente OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Crear directorios necesarios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "json"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "chunks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "embeddings"), exist_ok=True)
os.makedirs(TOXICITY_DIR, exist_ok=True)
os.makedirs(SAFETY_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

# ----------------- FUNCIONES DE CLASIFICACIÓN -----------------

def classify_lc50_toxicity(lc50_data):
    """
    Clasifica la toxicidad basada en los valores LC50.
    Devuelve un diccionario con la clasificación y el razonamiento.
    """
    if not lc50_data:
        return {"severity": "Unknown", "concern_level": "Unknown", "reasoning": "No LC50 data available"}
    
    lowest_lc50 = float('inf')
    lowest_lc50_unit = ""
    lowest_lc50_species = ""
    lowest_lc50_time = ""
    
    for item in lc50_data:
        try:
            value = float(item.get('valor', '0').replace(',', ''))
            unit = item.get('unidad', '').lower()
            if 'ppm' in unit:
                value = value * 4.0  # Conversión aproximada
                unit = 'mg/m³'
            if value < lowest_lc50 and 'mg/m' in unit:
                lowest_lc50 = value
                lowest_lc50_unit = unit
                lowest_lc50_species = item.get('especie', '')
                lowest_lc50_time = item.get('tiempo', '')
        except (ValueError, TypeError):
            continue
    
    if lowest_lc50 == float('inf'):
        return {"severity": "Unknown", "concern_level": "Unknown", "reasoning": "Could not determine LC50 value"}
    
    if lowest_lc50 <= 100:
        severity = "High"
        concern_level = "High"
        reasoning = f"LC50 value of {lowest_lc50} {lowest_lc50_unit} is very low, indicating high acute toxicity."
    elif lowest_lc50 <= 500:
        severity = "Moderate to High" 
        concern_level = "Moderate"
        reasoning = f"LC50 value of {lowest_lc50} {lowest_lc50_unit} indicates significant acute toxicity."
    elif lowest_lc50 <= 2000:
        severity = "Moderate"
        concern_level = "Low to Moderate"
        reasoning = f"LC50 value of {lowest_lc50} {lowest_lc50_unit} indicates moderate acute toxicity."
    elif lowest_lc50 <= 20000:
        severity = "Low"
        concern_level = "Low"
        reasoning = f"LC50 value of {lowest_lc50} {lowest_lc50_unit} indicates relatively low acute toxicity."
    else:
        severity = "Very Low"
        concern_level = "Very Low"
        reasoning = f"LC50 value of {lowest_lc50} {lowest_lc50_unit} indicates very low acute toxicity."
    
    return {
        "severity": severity,
        "concern_level": concern_level,
        "reasoning": reasoning,
        "lowest_lc50": lowest_lc50,
        "unit": lowest_lc50_unit,
        "species": lowest_lc50_species,
        "time": lowest_lc50_time
    }

# ----------------- FUNCIONES DE VALIDACIÓN Y NORMALIZACIÓN -----------------

def validate_toxicity_value(tipo, valor, unidad, especie="", via="", tiempo=""):
    try:
        val_float = float(valor.replace(",", ""))
        unidades_ld50 = ["mg/kg", "g/kg", "µg/kg", "ml/kg"]
        unidades_lc50 = ["mg/l", "mg/L", "ppm", "mg/m3", "mg/m³", "µg/l", "µg/L", "g/m3"]
        if tipo == "LD50":
            if not any(u.lower() in unidad.lower() for u in unidades_ld50):
                return False
            if "mg/kg" in unidad.lower() and (val_float < 0.1 or val_float > 20000):
                return False
            if "g/kg" in unidad.lower() and (val_float < 0.0001 or val_float > 20):
                return False
        elif tipo == "LC50":
            if not any(u.lower() in unidad.lower() for u in unidades_lc50):
                return False
            if "ppm" in unidad.lower() and (val_float < 0.1 or val_float > 2000000):
                return False
            if "mg/l" in unidad.lower() and (val_float < 0.001 or val_float > 1000):
                return False
            if "mg/m3" in unidad.lower() or "mg/m³" in unidad.lower():
                if (val_float < 0.001 or val_float > 2000000):
                    return False
        return True
    except:
        return False

def normalize_and_deduplicate_values(values, tipo):
    if not values:
        return []
    normalized_values = []
    for item in values:
        if tipo == "LD50" and validate_toxicity_value("LD50", item.get("valor", "0"), item.get("unidad", ""),
                                                  item.get("especie", ""), item.get("vía", "")):
            unit = item.get("unidad", "").lower()
            if "mg/kg" in unit:
                unit = "mg/kg"
            elif "g/kg" in unit:
                unit = "g/kg"
            elif "µg/kg" in unit:
                unit = "µg/kg"
            normalized_item = {
                "valor": item.get("valor", "").replace(",", "."),
                "unidad": unit,
                "especie": item.get("especie", "").capitalize() if item.get("especie") else "",
                "vía": item.get("vía", "").lower() if item.get("vía") else ""
            }
            if normalized_item["vía"] == "ip":
                normalized_item["vía"] = "intraperitoneal"
            elif normalized_item["vía"] == "iv":
                normalized_item["vía"] = "intravenous"
            elif normalized_item["vía"] == "sc":
                normalized_item["vía"] = "subcutaneous"
            is_duplicate = False
            for existing in normalized_values:
                try:
                    value_existing = float(existing["valor"])
                    value_current = float(normalized_item["valor"])
                    if (abs(value_existing - value_current) / max(value_existing, value_current) < 0.05 and
                        existing["unidad"] == normalized_item["unidad"] and
                        existing["especie"] == normalized_item["especie"]):
                        is_duplicate = True
                        if not existing["vía"] and normalized_item["vía"]:
                            existing["vía"] = normalized_item["vía"]
                        break
                except (ValueError, ZeroDivisionError):
                    continue
            if not is_duplicate:
                normalized_values.append(normalized_item)
        elif tipo == "LC50" and validate_toxicity_value("LC50", item.get("valor", "0"), item.get("unidad", ""),
                                                    item.get("especie", ""), item.get("via", ""), item.get("tiempo", "")):
            unit = item.get("unidad", "").lower()
            if "ppm" in unit:
                unit = "ppm"
            elif "mg/l" in unit or "mg/L" in unit:
                unit = "mg/L"
            elif "mg/m3" in unit or "mg/m³" in unit:
                unit = "mg/m³"
            normalized_item = {
                "valor": item.get("valor", "").replace(",", "."),
                "unidad": unit,
                "especie": item.get("especie", "").capitalize() if item.get("especie") else "",
                "tiempo": item.get("tiempo", ""),
                "via": "inhalation" if not item.get("via") else item.get("via", "").lower()
            }
            is_duplicate = False
            for existing in normalized_values:
                try:
                    value_existing = float(existing["valor"])
                    value_current = float(normalized_item["valor"])
                    if (abs(value_existing - value_current) / max(value_existing, value_current) < 0.05 and
                        existing["unidad"] == normalized_item["unidad"] and
                        existing["especie"] == normalized_item["especie"]):
                        is_duplicate = True
                        if not existing["tiempo"] and normalized_item["tiempo"]:
                            existing["tiempo"] = normalized_item["tiempo"]
                        break
                except (ValueError, ZeroDivisionError):
                    continue
            if not is_duplicate:
                normalized_values.append(normalized_item)
    if tipo == "LD50":
        normalized_values = sorted(normalized_values, key=lambda x: (1 if x.get("especie") else 0) + (1 if x.get("vía") else 0), reverse=True)
    else:
        normalized_values = sorted(normalized_values, key=lambda x: (1 if x.get("especie") else 0) + (1 if x.get("tiempo") else 0) + (1 if x.get("via") else 0), reverse=True)
    return normalized_values

# ----------------- FUNCIONES DE EXTRACCIÓN DE DATOS -----------------

def get_toxicity_data_from_pubchem(cid):
    results = {"LD50": [], "LC50": []}
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Toxicity"
        logger.info(f"Consultando API de PubChem para toxicidad: {url}")
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()

            def extract_toxicity_data(section):
                if isinstance(section, dict):
                    if 'TOCHeading' in section and 'Toxicity' in section.get('TOCHeading', ''):
                        if 'Section' in section:
                            for subsection in section['Section']:
                                extract_toxicity_data(subsection)
                    if 'TOCHeading' in section and 'LD50' in section.get('TOCHeading', ''):
                        if 'Information' in section:
                            for info in section['Information']:
                                if 'Value' in info and 'StringWithMarkup' in info['Value']:
                                    for item in info['Value']['StringWithMarkup']:
                                        if 'String' in item:
                                            text = item['String']
                                            ld50_match = re.search(r'([\d\.,]+)\s*(mg\/kg|g\/kg|µg\/kg)', text)
                                            if ld50_match:
                                                valor, unidad = ld50_match.groups()
                                                especie = "Unknown"
                                                via = "Unknown"
                                                especie_match = re.search(r'(rat|mouse|rabbit)', text, re.IGNORECASE)
                                                if especie_match:
                                                    especie = especie_match.group(1).capitalize()
                                                via_match = re.search(r'(oral|dermal|intraperitoneal)', text, re.IGNORECASE)
                                                if via_match:
                                                    via = via_match.group(1).lower()
                                                results["LD50"].append({
                                                    "valor": valor.replace(",", ""),
                                                    "unidad": unidad.lower(),
                                                    "especie": especie,
                                                    "vía": via,
                                                    "source": "PubChem API"
                                                })
                    if 'TOCHeading' in section and 'LC50' in section.get('TOCHeading', ''):
                        if 'Information' in section:
                            for info in section['Information']:
                                if 'Value' in info and 'StringWithMarkup' in info['Value']:
                                    for item in info['Value']['StringWithMarkup']:
                                        if 'String' in item:
                                            text = item['String']
                                            lc50_match = re.search(r'([\d\.,]+)\s*(mg\/l|ppm|mg\/m3|mg\/m³)', text, re.IGNORECASE)
                                            if lc50_match:
                                                valor, unidad = lc50_match.groups()
                                                especie = "Unknown"
                                                tiempo = ""
                                                via = "inhalation"
                                                especie_match = re.search(r'(rat|mouse|rabbit|fish|daphnia)', text, re.IGNORECASE)
                                                if especie_match:
                                                    especie = especie_match.group(1).capitalize()
                                                tiempo_match = re.search(r'(\d+)\s*(?:h|hr|hour|hours)', text, re.IGNORECASE)
                                                if tiempo_match:
                                                    tiempo = f"{tiempo_match.group(1)} hr"
                                                results["LC50"].append({
                                                    "valor": valor.replace(",", ""),
                                                    "unidad": unidad.lower(),
                                                    "especie": especie,
                                                    "tiempo": tiempo,
                                                    "via": via,
                                                    "source": "PubChem API"
                                                })
                    if 'Section' in section:
                        for subsection in section['Section']:
                            extract_toxicity_data(subsection)

            if 'Record' in data and 'Section' in data['Record']:
                for section in data['Record']['Section']:
                    extract_toxicity_data(section)
            results["LD50"] = normalize_and_deduplicate_values(results["LD50"], "LD50")
            results["LC50"] = normalize_and_deduplicate_values(results["LC50"], "LC50")
            logger.info(f"Datos de toxicidad extraídos de PubChem API: {len(results['LD50'])} valores LD50, {len(results['LC50'])} valores LC50")
        return results
    except Exception as e:
        logger.error(f"Error al extraer datos de toxicidad desde PubChem API: {str(e)}")
        return results

def extract_ld50_lc50_direct(cas, cid):
    logger.info(f"Iniciando extracción de LD50/LC50 para CAS {cas} (CID: {cid})")
    results = {"LD50": [], "LC50": []}
    api_results = get_toxicity_data_from_pubchem(cid)
    results["LD50"].extend(api_results["LD50"])
    results["LC50"].extend(api_results["LC50"])
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading=Toxicity+Data"
    print(f"Consultando URL: {url}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            raw_data = response.text
            ld50_matches = re.findall(r'LD50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/kg|g\/kg|µg\/kg|ml\/kg|ppm)', raw_data, re.IGNORECASE)
            validated_ld50 = []
            for match in ld50_matches:
                valor = match[0].replace(",", "")
                unidad = match[1]
                context_start = raw_data.find(match[0]) - 150 if raw_data.find(match[0]) > 150 else 0
                context_end = raw_data.find(match[0]) + 300
                context = raw_data[context_start:context_end]
                especie = ""
                via = ""
                especie_match = re.search(r'(?:rat|mouse|rabbit|guinea\s*pig|daphnia|fish|trout|zebrafish)s?', context, re.IGNORECASE)
                if especie_match:
                    especie = especie_match.group(0).capitalize()
                via_match = re.search(r'\b(?:oral|dermal|intraperitoneal|subcutaneous|intravenous|ip|iv|sc)\b', context, re.IGNORECASE)
                if via_match:
                    via = via_match.group(0).lower()
                    if via == "ip": via = "intraperitoneal"
                    if via == "iv": via = "intravenous"
                    if via == "sc": via = "subcutaneous"
                if validate_toxicity_value("LD50", match[0], unidad, especie, via):
                    normalized_unit = unidad.lower()
                    if "mg/kg" in normalized_unit:
                        normalized_unit = "mg/kg"
                    elif "g/kg" in normalized_unit:
                        normalized_unit = "g/kg"
                    if not any(abs(float(existing["valor"]) - float(match[0].replace(",", ""))) / float(match[0].replace(",", "")) < 0.05 for existing in validated_ld50):
                        validated_ld50.append({
                            "valor": match[0].replace(",", ""),
                            "unidad": normalized_unit,
                            "especie": especie,
                            "vía": via,
                            "source": "Toxicity Data"
                        })
            validated_lc50 = []
            lc50_patterns = [
                r'LC50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³|µg\/[lL]|g\/m3)(?:\s*\/\s*(\d+)\s*h(?:ours?|r)?)?',
                r'LC 50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³|µg\/[lL]|g\/m3)(?:\s*\/\s*(\d+)\s*h(?:ours?|r)?)?',
                r'inhalation\s+LC50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³)(?:\s*\/\s*(\d+)\s*h(?:ours?|r)?)?',
                r'rat\s+LC50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³)(?:\s*\/\s*(\d+)\s*h(?:ours?|r)?)?',
                r'LC50\s*\([^)]+\)\s*=\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³|µg\/[lL]|g\/m3)(?:\/(\d+)h)?'
            ]
            for pattern in lc50_patterns:
                matches = re.findall(pattern, raw_data, re.IGNORECASE)
                for match in matches:
                    if len(match) < 2:
                        continue
                    valor = match[0].replace(",", "")
                    unidad_completa = match[1]
                    tiempo = ""
                    if len(match) > 2 and match[2]:
                        tiempo = f"{match[2]} hr"
                    context_start = raw_data.find(match[0]) - 150 if raw_data.find(match[0]) > 150 else 0
                    context_end = raw_data.find(match[0]) + 300
                    context = raw_data[context_start:context_end]
                    especie = ""
                    via = "inhalation"
                    especie_match = re.search(r'(?:rat|mouse|rabbit|fish|daphnia|shrimp)', context, re.IGNORECASE)
                    if especie_match:
                        especie = especie_match.group(0).capitalize()
                    if not tiempo:
                        tiempo_match = re.search(r'(?:for|during)?\s*(\d+)\s*(?:h|hr|hour|hours)', context, re.IGNORECASE)
                        if tiempo_match:
                            tiempo = f"{tiempo_match.group(1)} hr"
                    if validate_toxicity_value("LC50", valor, unidad_completa, especie, via, tiempo):
                        normalized_unit = unidad_completa.lower()
                        if "ppm" in normalized_unit:
                            normalized_unit = "ppm"
                        elif "mg/l" in normalized_unit or "mg/L" in normalized_unit:
                            normalized_unit = "mg/L"
                        elif "mg/m3" in normalized_unit or "mg/m³" in normalized_unit:
                            normalized_unit = "mg/m³"
                        if not any(abs(float(existing["valor"]) - float(valor)) / float(valor) < 0.05 for existing in validated_lc50):
                            validated_lc50.append({
                                "valor": valor,
                                "unidad": normalized_unit,
                                "especie": especie,
                                "tiempo": tiempo,
                                "via": via,
                                "source": "Toxicity Data"
                            })
            results["LD50"].extend(validated_ld50)
            results["LC50"].extend(validated_lc50)
            results["LD50"] = normalize_and_deduplicate_values(results["LD50"], "LD50")
            results["LC50"] = normalize_and_deduplicate_values(results["LC50"], "LC50")
            logger.info(f"Datos extraídos de Toxicity Data: {len(validated_ld50)} valores LD50, {len(validated_lc50)} valores LC50")
    except Exception as e:
        logger.error(f"Error consultando Toxicity Data: {str(e)}")
    urls = [
        {"name": "LD50", "url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?name=LD50"},
        {"name": "LC50", "url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?name=LC50"}
    ]
    for url_info in urls:
        try:
            url = url_info["url"]
            print(f"Consultando URL: {url}")
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                with open(f"{cas}_{url_info['name']}_response.json", 'w') as f:
                    json.dump(data, f, indent=2)
                raw_text = json.dumps(data)
                if url_info["name"] == "LD50":
                    ld50_matches = re.findall(r'LD50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/kg|g\/kg|µg\/kg|ml\/kg|ppm)', raw_text, re.IGNORECASE)
                    validated_matches = []
                    for match in ld50_matches:
                        valor = match[0].replace(",", "")
                        unidad = match[1]
                        context_start = raw_text.find(match[0]) - 150 if raw_text.find(match[0]) > 150 else 0
                        context_end = raw_text.find(match[0]) + 300
                        context = raw_text[context_start:context_end]
                        especie = ""
                        via = ""
                        especie_match = re.search(r'(?:rat|mouse|rabbit|guinea\s*pig)', context, re.IGNORECASE)
                        if especie_match:
                            especie = especie_match.group(0).capitalize()
                        via_match = re.search(r'\b(?:oral|dermal|intraperitoneal|subcutaneous|intravenous|ip|iv|sc)\b', context, re.IGNORECASE)
                        if via_match:
                            via = via_match.group(0).lower()
                            if via == "ip": via = "intraperitoneal"
                            if via == "iv": via = "intravenous"
                            if via == "sc": via = "subcutaneous"
                        if validate_toxicity_value("LD50", valor, unidad, especie, via):
                            validated_matches.append({
                                "valor": valor,
                                "unidad": unidad.lower(),
                                "especie": especie,
                                "vía": via,
                                "source": f"Search URL {url_info['name']}"
                            })
                    results["LD50"].extend(validated_matches)
                    logger.info(f"Datos LD50 extraídos de URL {url_info['name']}: {len(validated_matches)} valores válidos")
                elif url_info["name"] == "LC50":
                    lc50_patterns = [
                        r'LC50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³|µg\/[lL]|g\/m3)(?:\s*\/\s*(\d+)\s*h(?:ours?|r)?)?',
                        r'LC 50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³|µg\/[lL]|g\/m3)(?:\s*\/\s*(\d+)\s*h(?:ours?|r)?)?',
                        r'inhalation\s+LC50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³)(?:\s*\/\s*(\d+)\s*h(?:ours?|r)?)?',
                        r'rat\s+LC50\s*(?:\(|:|=|\s)\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³)(?:\s*\/\s*(\d+)\s*h(?:ours?|r)?)?',
                        r'LC50\s*\([^)]+\)\s*=\s*([\d\.,]+)\s*(mg\/[lL]|ppm|mg\/m3|mg\/m³|µg\/[lL]|g\/m3)(?:\/(\d+)h)?'
                    ]
                    validated_matches = []
                    for pattern in lc50_patterns:
                        matches = re.findall(pattern, raw_text, re.IGNORECASE)
                        for match in matches:
                            if len(match) < 2:
                                continue
                            valor = match[0].replace(",", "")
                            unidad = match[1]
                            tiempo = ""
                            if len(match) > 2 and match[2]:
                                tiempo = f"{match[2]} hr"
                            context_start = raw_text.find(match[0]) - 150 if raw_text.find(match[0]) > 150 else 0
                            context_end = raw_text.find(match[0]) + 300
                            context = raw_text[context_start:context_end]
                            especie = ""
                            via = "inhalation"
                            especie_match = re.search(r'(?:rat|mouse|rabbit|fish|daphnia|shrimp)', context, re.IGNORECASE)
                            if especie_match:
                                especie = especie_match.group(0).capitalize()
                            if not tiempo:
                                tiempo_match = re.search(r'(?:for|during)?\s*(\d+)\s*(?:h|hr|hour|hours)', context, re.IGNORECASE)
                                if tiempo_match:
                                    tiempo = f"{tiempo_match.group(1)} hr"
                            if validate_toxicity_value("LC50", valor, unidad, especie, via, tiempo):
                                validated_matches.append({
                                    "valor": valor,
                                    "unidad": unidad.lower(),
                                    "especie": especie,
                                    "tiempo": tiempo,
                                    "via": via,
                                    "source": f"Search URL {url_info['name']}"
                                })
                    results["LC50"].extend(validated_matches)
                    logger.info(f"Datos LC50 extraídos de URL {url_info['name']}: {len(validated_matches)} valores válidos")
                results["LD50"] = normalize_and_deduplicate_values(results["LD50"], "LD50")
                results["LC50"] = normalize_and_deduplicate_values(results["LC50"], "LC50")
            time.sleep(API_SLEEP_TIME)
        except Exception as e:
            logger.error(f"Error consultando {url_info['name']}: {str(e)}")
    for item in results["LD50"] + results["LC50"]:
        item["cas"] = cas
        item["cid"] = cid
        item["extraction_date"] = time.strftime("%Y-%m-%d")
    logger.info(f"Extracción finalizada. Datos validados: {len(results['LD50'])} valores LD50, {len(results['LC50'])} valores LC50")
    for i, item in enumerate(results['LD50']):
        logger.info(f"LD50 #{i+1}: {item.get('valor', 'N/A')} {item.get('unidad', 'N/A')} (Especie: {item.get('especie', 'N/A')}, Vía: {item.get('vía', 'N/A')})")
    for i, item in enumerate(results['LC50']):
        logger.info(f"LC50 #{i+1}: {item.get('valor', 'N/A')} {item.get('unidad', 'N/A')} (Especie: {item.get('especie', 'N/A')}, Tiempo: {item.get('tiempo', 'N/A')}, Vía: {item.get('via', 'N/A')})")
    print(f"Total encontrado: {len(results['LD50'])} valores LD50 y {len(results['LC50'])} valores LC50")
    return results

# ----------------- CLASE CHEMICALSAFETYANALYZER -----------------

class ChemicalSafetyAnalyzer:
    def __init__(self, output_dir=OUTPUT_DIR, collection_name=CHROMADB_COLLECTION, persist_dir=CHROMA_PERSIST_DIR):
        self.output_dir = output_dir
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.cas_to_cid_map = {}
        try:
            self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            logger.info(f"Cliente ChromaDB inicializado con persistencia en {self.persist_dir}")
        except Exception as e:
            logger.error(f"Error al inicializar ChromaDB: {str(e)}")
            raise
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Colección ChromaDB '{collection_name}' inicializada")
        except Exception as e:
            logger.error(f"Error al inicializar colección ChromaDB: {str(e)}")
            raise

    def get_embedding(self, text, model="text-embedding-3-small"):
        if not text or len(text.strip()) == 0:
            return [0.0] * 1536
        if len(text) > 8000:
            text = text[:8000]
        text = text.replace("\n", " ")
        try:
            response = openai_client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error al obtener embedding: {str(e)}")
            return [0.0] * 1536

    def get_cid_from_cas(self, cas):
        try:
            if cas in self.cas_to_cid_map:
                return self.cas_to_cid_map[cas]
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/xref/RN/{cas}/cids/JSON"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
                    cid = data['IdentifierList']['CID'][0]
                    self.cas_to_cid_map[cas] = cid
                    return cid
            logger.warning(f"No se encontró CID para el CAS {cas}")
            return None
        except Exception as e:
            logger.error(f"Error al obtener CID para CAS {cas}: {str(e)}")
            return None

    def get_compound_data(self, cid):
        compound_data = {}
        try:
            url_props = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularFormula,MolecularWeight,XLogP,IUPAC,InChI,InChIKey,CanonicalSMILES,HBondDonorCount,HBondAcceptorCount,RotatableBondCount/JSON"
            response = requests.get(url_props)
            if response.status_code == 200:
                data = response.json()
                if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                    compound_data['basic_properties'] = data['PropertyTable']['Properties'][0]
            time.sleep(API_SLEEP_TIME)
            categories = {
                'chemical_properties': 'Chemical+and+Physical+Properties',
                'toxicity': 'Toxicity',
                'safety': 'Safety+and+Hazards',
                'uses': 'Use+and+Manufacturing',
                'ghs_classification': 'GHS+Classification'
            }
            for key, category in categories.items():
                url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON?heading={category}"
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    compound_data[key] = data
                json_path = os.path.join(self.output_dir, "json", f"{cid}_{key}.json")
                with open(json_path, 'w') as f:
                    json.dump(data if response.status_code == 200 else {}, f)
                time.sleep(API_SLEEP_TIME)
            return compound_data
        except Exception as e:
            logger.error(f"Error al obtener datos para CID {cid}: {str(e)}")
            return compound_data

    def extract_text_from_section(self, section_data):
        texts = []
        def extract_from_section(section, path=""):
            if isinstance(section, dict):
                if 'TOCHeading' in section:
                    heading = section['TOCHeading']
                    if any(keyword in heading.lower() for keyword in ['toxic', 'hazard', 'safety', 'ld50', 'lc50', 'risk']):
                        texts.append(f"\n--- {heading} ---\n")
                if 'Value' in section and 'StringWithMarkup' in section['Value']:
                    for item in section['Value']['StringWithMarkup']:
                        if 'String' in item:
                            text = item['String'].replace('\u000A', '\n')
                            texts.append(text)
                if 'Description' in section:
                    description = section['Description']
                    if len(description) > 20 and any(keyword in description.lower() for keyword in ['toxic', 'hazard', 'safety', 'risk', 'ld50', 'lc50']):
                        texts.append(f"Description: {description}")
                if 'Reference' in section:
                    ref_texts = []
                    if isinstance(section['Reference'], list):
                        for ref in section['Reference']:
                            if isinstance(ref, str):
                                ref_texts.append(ref)
                            elif isinstance(ref, dict):
                                if 'SourceName' in ref:
                                    ref_texts.append(ref['SourceName'])
                                elif 'Name' in ref:
                                    ref_texts.append(ref['Name'])
                                elif 'URL' in ref:
                                    ref_texts.append(ref['URL'])
                                elif 'Description' in ref:
                                    ref_texts.append(ref['Description'])
                    elif isinstance(section['Reference'], str):
                        ref_texts.append(section['Reference'])
                    if ref_texts:
                        texts.append(f"Reference: {'; '.join(ref_texts)}")
                if 'Section' in section:
                    for i, subsection in enumerate(section['Section']):
                        extract_from_section(subsection, f"{path}.Section[{i}]")
                if 'Information' in section:
                    for i, info in enumerate(section['Information']):
                        extract_from_section(info, f"{path}.Information[{i}]")
                for key in ['Record', 'Data', 'Compound']:
                    if key in section:
                        extract_from_section(section[key], f"{path}.{key}")
            elif isinstance(section, list):
                for i, item in enumerate(section):
                    extract_from_section(item, f"{path}[{i}]")
        if 'Record' in section_data and 'Section' in section_data['Record']:
            extract_from_section(section_data['Record'], "Record")
        else:
            extract_from_section(section_data, "root")
        unique_texts = []
        for text in texts:
            if text not in unique_texts:
                unique_texts.append(text)
        result = "\n".join(unique_texts)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result

    def split_text_into_chunks(self, text, max_chunk_size=MAX_CHUNK_SIZE):
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if len(paragraph) > max_chunk_size:
                sentences = paragraph.replace(". ", ".\n").split("\n")
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                        current_chunk = current_chunk + "\n\n" + sentence if current_chunk else sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                    current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
                else:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def create_document_for_compound(self, cas, cid, compound_data):
        document = {
            "cas": cas,
            "cid": cid,
            "name": compound_data.get('basic_properties', {}).get('IUPACName', f"Compound {cid}"),
            "formula": compound_data.get('basic_properties', {}).get('MolecularFormula', ''),
            "weight": compound_data.get('basic_properties', {}).get('MolecularWeight', ''),
            "xlogp": compound_data.get('basic_properties', {}).get('XLogP', ''),
            "smiles": compound_data.get('basic_properties', {}).get('CanonicalSMILES', ''),
        }
        text_sections = {}
        for key in ['chemical_properties', 'toxicity', 'safety', 'uses', 'ghs_classification']:
            if key in compound_data:
                text_sections[key] = self.extract_text_from_section(compound_data[key])
        full_text = f"CAS: {cas}\nCID: {cid}\n"
        full_text += f"Name: {document['name']}\nFormula: {document['formula']}\nMolecular Weight: {document['weight']}\nXLogP: {document['xlogp']}\nSMILES: {document['smiles']}\n\n"
        for key, text in text_sections.items():
            title = key.replace('_', ' ').title()
            full_text += f"--- {title} ---\n{text}\n\n"
        document['text'] = full_text
        document['sections'] = text_sections
        return document

    def extract_toxicity_sections(self, chunk):
        """
        Extrae secciones relevantes sobre toxicidad y genotoxicidad de un chunk de texto.
        Ahora ampliado para capturar menciones de genotoxicidad y carcinogenicidad.
        """
        if len(chunk) < 500:
            return chunk
        
        relevant_text = []
        toxicity_patterns = [
            r"(?i)--- Toxicity ---.*?(?=---|\Z)",
            r"(?i)--- Safety and Hazards ---.*?(?=---|\Z)",
            r"(?i)--- GHS Classification ---.*?(?=---|\Z)",
            r"(?i)--- Ecological Information ---.*?(?=---|\Z)",
            r"(?i)TOCHeading.*?Toxicity.*?(?=TOCHeading|\Z)",
            r"(?i)TOCHeading.*?Hazard.*?(?=TOCHeading|\Z)",
            r"(?i)TOCHeading.*?Safety.*?(?=TOCHeading|\Z)",
            r"(?i)TOCHeading.*?LD50.*?(?=TOCHeading|\Z)",
            r"(?i)TOCHeading.*?LC50.*?(?=TOCHeading|\Z)",
            r"(?i)TOCHeading.*?Acute Effects.*?(?=TOCHeading|\Z)",
            r"(?i)TOCHeading.*?Toxicity Data.*?(?=TOCHeading|\Z)",
            r"(?i)LD50.*?mg/kg.*?(?=\n\n|\Z)",
            r"(?i)LC50.*?ppm.*?(?=\n\n|\Z)",
            r"(?i)LD50.*?g/kg.*?(?=\n\n|\Z)",
            r"(?i)LC50.*?mg/L.*?(?=\n\n|\Z)",
            r"(?i)LD50.*?µg/kg.*?(?=\n\n|\Z)",
            r"(?i)Toxicity[:\s].*?(?=\n\n|\Z)",
            r"(?i)Hazard.*?(?=\n\n|\Z)",
            r"(?i)Carcinogen.*?(?=\n\n|\Z)",
            r"(?i)Mutagen.*?(?=\n\n|\Z)",
            r"(?i)toxic effect.*?(?=\n\n|\Z)",
            r"(?i)health hazard.*?(?=\n\n|\Z)",
            r"(?i)risk phrase.*?(?=\n\n|\Z)",
            r"(?i)safety phrase.*?(?=\n\n|\Z)",
            r"(?i)\bLD50\b.*?(?=\n\n|\Z)",
            r"(?i)\bLC50\b.*?(?=\n\n|\Z)",
            r"(?i)\bEC50\b.*?(?=\n\n|\Z)",
            r"(?i)\bTLV\b.*?(?=\n\n|\Z)",
            r"(?i)\bPEL\b.*?(?=\n\n|\Z)",
            r"(?i)\bIDLH\b.*?(?=\n\n|\Z)",
            r'(?i)"TOCHeading"\s*:\s*"Toxicity Data".*?(?="TOCHeading"|\Z)',
            r'(?i)"TOCHeading"\s*:\s*"Acute Effects".*?(?="TOCHeading"|\Z)',
            r'(?i)"String"\s*:\s*"LD50.*?(?=},|\])',
            r'(?i)"String"\s*:\s*"LC50.*?(?=},|\])',
            r'(?i)"TOCHeading"\s*:\s*"Mutagen.*?(?="TOCHeading"|\Z)',
            r'(?i)"TOCHeading"\s*:\s*"Carcinogen.*?(?="TOCHeading"|\Z)',
            r'(?i)"TOCHeading"\s*:\s*"Genotoxic.*?(?="TOCHeading"|\Z)',
            r'(?i)"String"\s*:\s*".*?genotox.*?(?=},|\])',
            r'(?i)"String"\s*:\s*".*?mutagen.*?(?=},|\])',
            r'(?i)"String"\s*:\s*".*?carcinog.*?(?=},|\])',
            r'(?i)"String"\s*:\s*".*?cancer.*?(?=},|\])'
        ]
        for pattern in toxicity_patterns:
            matches = re.findall(pattern, chunk, re.DOTALL)
            for match in matches:
                if match and len(match.strip()) > 10:
                    if not any(self.similarity(match, existing) > 0.8 for existing in relevant_text):
                        relevant_text.append(match.strip())
        if not relevant_text:
            genotoxicity_terms = [
                r"genotox", r"mutagen", r"carcinog", r"cancer", r"tumor", 
                r"dna\s+damage", r"chromosome\s+aberration", r"micronucleus", 
                r"ames\s+test", r"genetic\s+toxicity", r"clastogenic"
            ]
            for term in genotoxicity_terms:
                for match in re.finditer(term, chunk, re.IGNORECASE):
                    start = max(0, match.start() - 200)
                    end = min(len(chunk), match.end() + 200)
                    context = chunk[start:end]
                    relevant_text.append(context)
        if not relevant_text:
            return chunk
        return "\n\n".join(relevant_text)

    def similarity(self, text1, text2):
        if not text1 or not text2:
            return 0
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        if not words1 or not words2:
            return 0
        common_words = words1.intersection(words2)
        return len(common_words) / min(len(words1), len(words2))

    def process_toxicity_data_workflow_optimized(self, cas):
        try:
            logger.info(f"Iniciando flujo de trabajo optimizado para datos de toxicidad de CAS {cas}")
            cid = self.get_cid_from_cas(cas)
            if not cid:
                logger.warning(f"No se pudo obtener CID para CAS {cas}")
                return {"success": False, "error": "CID no encontrado"}
            
            # 1. Extraer valores LD50/LC50 con método directo
            logger.info(f"Extrayendo valores LD50/LC50 con método directo para CAS {cas}")
            ld50_lc50_data = extract_ld50_lc50_direct(cas, cid)
            
            # 2. Obtener todos los chunks de texto disponibles para un análisis completo
            logger.info(f"Obteniendo todos los chunks de texto para CAS {cas}")
            all_chunks = []
            results = self.collection.get(where={"cas": cas}, limit=1000)
            if results["ids"]:
                all_chunks = results["documents"]
                logger.info(f"Recuperados {len(all_chunks)} chunks existentes de ChromaDB para CAS {cas}")
            if not all_chunks:
                logger.info(f"No se encontraron chunks en ChromaDB, obteniendo datos de PubChem para CAS {cas}")
                compound_data = self.get_compound_data(cid)
                document = self.create_document_for_compound(cas, cid, compound_data)
                all_chunks = self.split_text_into_chunks(document['text'])
                logger.info(f"Obtenidos {len(all_chunks)} chunks de texto de PubChem para CAS {cas}")
            
            # 3. Preparar datos estructurados de toxicidad
            toxicity_text = f"# Datos de Toxicidad para Compuesto CAS {cas} (CID {cid})\n\n"
            if ld50_lc50_data["LD50"]:
                toxicity_text += "## LD50 (Dosis Letal 50%)\n\n"
                for item in ld50_lc50_data["LD50"]:
                    toxicity_text += f"- **Valor**: {item.get('valor')} {item.get('unidad')}\n"
                    if item.get('especie'):
                        toxicity_text += f"  **Especie**: {item.get('especie')}\n"
                    if item.get('vía'):
                        toxicity_text += f"  **Vía**: {item.get('vía')}\n"
                    toxicity_text += "\n"
            if ld50_lc50_data["LC50"]:
                toxicity_text += "## LC50 (Concentración Letal 50%)\n\n"
                for item in ld50_lc50_data["LC50"]:
                    toxicity_text += f"- **Valor**: {item.get('valor')} {item.get('unidad')}\n"
                    if item.get('especie'):
                        toxicity_text += f"  **Especie**: {item.get('especie')}\n"
                    if item.get('tiempo'):
                        toxicity_text += f"  **Tiempo**: {item.get('tiempo')}\n"
                    if item.get('via'):
                        toxicity_text += f"  **Vía**: {item.get('via')}\n"
                    toxicity_text += "\n"
            
            # 4. Buscar menciones de genotoxicidad en todos los chunks
            genotoxicity_terms = [
                r"genotox", r"mutagen", r"carcinog", r"cancer", r"tumor", 
                r"dna\s+damage", r"chromosome\s+aberration", r"micronucleus", 
                r"ames\s+test", r"genetic\s+toxicity", r"clastogenic"
            ]
            genotoxicity_mentions = []
            for i, chunk in enumerate(all_chunks):
                for term in genotoxicity_terms:
                    for match in re.finditer(term, chunk, re.IGNORECASE):
                        start = max(0, match.start() - 150)
                        end = min(len(chunk), match.end() + 150)
                        context = chunk[start:end]
                        genotoxicity_mentions.append({
                            "term": match.group(),
                            "context": context,
                            "chunk_index": i
                        })
            
            if genotoxicity_mentions:
                toxicity_text += "## Menciones de Genotoxicidad/Carcinogenicidad\n\n"
                for i, mention in enumerate(genotoxicity_mentions[:10]):
                    toxicity_text += f"- **Mención {i+1}**: '{mention['term']}'\n"
                    toxicity_text += f"  **Contexto**: \"...{mention['context']}...\"\n\n"
            
            # 5. Guardar y persistir los datos
            result = {
                "cas": cas,
                "cid": cid,
                "raw_data": ld50_lc50_data,
                "genotoxicity_mentions": genotoxicity_mentions,
                "success": True
            }
            output_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_data_raw.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            # 6. Actualizar ChromaDB con los datos de toxicidad
            chunk_id = f"cas-{cas}-toxicity-data"
            embedding_text = toxicity_text[:8000]
            try:
                embedding = self.get_embedding(embedding_text)
                metadata = {
                    "cas": cas,
                    "cid": str(cid),
                    "chunk_type": "toxicity_data",
                    "source": "direct_extraction",
                    "ld50_count": len(ld50_lc50_data["LD50"]),
                    "lc50_count": len(ld50_lc50_data["LC50"]),
                    "genotoxicity_mentions": len(genotoxicity_mentions),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                existing_chunk = self.collection.get(ids=[chunk_id], include=["metadatas"])
                if existing_chunk["ids"]:
                    logger.info(f"Actualizando chunk de toxicidad para CAS {cas}")
                    self.collection.update(
                        ids=[chunk_id],
                        documents=[toxicity_text[:32000]],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
                else:
                    logger.info(f"Creando nuevo chunk de toxicidad para CAS {cas}")
                    self.collection.add(
                        ids=[chunk_id],
                        documents=[toxicity_text[:32000]],
                        embeddings=[embedding],
                        metadatas=[metadata]
                    )
            except Exception as e:
                logger.warning(f"Error al almacenar en ChromaDB: {str(e)} - Continuando sin almacenar")
            
            # 7. Preparar todos los chunks para GPT
            combined_text = toxicity_text + "\n\n## Texto completo de todas las fuentes\n\n"
            max_combined_length = 24000  # Límite total
            remaining_length = max_combined_length - len(combined_text)
            chunks_for_gpt = []
            current_length = 0
            for chunk in all_chunks:
                processed_chunk = self.extract_toxicity_sections(chunk)
                if len(processed_chunk) + current_length <= remaining_length:
                    chunks_for_gpt.append(processed_chunk)
                    current_length += len(processed_chunk)
                else:
                    remaining_space = remaining_length - current_length
                    if remaining_space > 200:
                        chunks_for_gpt.append(processed_chunk[:remaining_space])
                    break
            if chunks_for_gpt:
                combined_text += "".join([f"\n\n--- Chunk de texto ---\n{chunk}\n" for chunk in chunks_for_gpt])
            
            # 8. Enviar a GPT para análisis
            prompt = f"""
Analiza la siguiente información sobre datos de toxicidad para el compuesto con CAS {cas}:

{combined_text}

INSTRUCCIONES:
1. Analiza TODA la información proporcionada (datos estructurados LD50/LC50 y todos los chunks de texto).
2. Busca ESPECÍFICAMENTE cualquier mención directa o indirecta de:
   - Genotoxicidad o mutagenicidad
   - Carcinogenicidad o potencial cancerígeno
   - Daño al ADN o aberraciones cromosómicas
   - Resultados de pruebas de Ames o micronúcleos
   - Cualquier otra evidencia de potencial daño genético
3. Si no hay menciones directas de genotoxicidad, evalúa el potencial genotóxico basándote en los valores de LC50:
   - LC50 ≤ 100 mg/m³: Alto nivel de preocupación
   - LC50 ≤ 500 mg/m³: Nivel moderado de preocupación
   - LC50 ≤ 2000 mg/m³: Bajo nivel de preocupación
4. Emite una clasificación final siguiendo estas reglas:
   - Si existe evidencia clara de genotoxicidad/carcinogenicidad: CLASIFICACIÓN_GENOTOXICIDAD: POSITIVO
   - Si existe evidencia clara de ausencia de genotoxicidad: CLASIFICACIÓN_GENOTOXICIDAD: NEGATIVO
   - Si no hay suficientes datos para una evaluación concluyente: CLASIFICACIÓN_GENOTOXICIDAD: INDETERMINADO
   - Si no hay datos directos de genotoxicidad, pero se observan valores de LC50 muy bajos (≤ 100 mg/m³): CLASIFICACIÓN_GENOTOXICIDAD: PRECAUTORIO
5. IMPORTANTE: Prioriza siempre las menciones directas de genotoxicidad sobre las inferencias basadas en LC50.

Incluye en tu análisis:
- Resumen de hallazgos principales
- Evaluación detallada de evidencia de genotoxicidad/carcinogenicidad
- Análisis de los valores LD50/LC50 y su significado toxicológico
- Una conclusión final con la clasificación de genotoxicidad
"""
            try:
                logger.info(f"Consultando a {GPT_MODEL} para análisis de toxicidad del CAS {cas}")
                response = openai_client.chat.completions.create(
                    model=GPT_MODEL,
                    messages=[
                        {"role": "system", "content": "Eres un toxicólogo experto que proporciona análisis precisos y exhaustivos sobre datos de toxicidad química. Buscas minuciosamente cualquier indicio de genotoxicidad en toda la información proporcionada."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2000
                )
                analysis = response.choices[0].message.content.strip()
                # MODIFICACIÓN 2: Mejorar la extracción de la clasificación de genotoxicidad
                genotoxicity_classification = "INDETERMINADO"
                if "CLASIFICACIÓN_GENOTOXICIDAD:" in analysis:
                    for line in analysis.split('\n'):
                        if "CLASIFICACIÓN_GENOTOXICIDAD:" in line:
                            parts = line.split(':', 1)
                            if len(parts) > 1:
                                genotoxicity_classification = parts[1].strip().upper()
                                break
                if genotoxicity_classification == "INDETERMINADO":
                    if "CLASIFICACIÓN GENOTOXICIDAD:" in analysis:
                        for line in analysis.split('\n'):
                            if "CLASIFICACIÓN GENOTOXICIDAD:" in line:
                                parts = line.split(':', 1)
                                if len(parts) > 1:
                                    genotoxicity_classification = parts[1].strip().upper()
                                    break
                    elif "CLASIFICACIÓN DE GENOTOXICIDAD:" in analysis:
                        for line in analysis.split('\n'):
                            if "CLASIFICACIÓN DE GENOTOXICIDAD:" in line:
                                parts = line.split(':', 1)
                                if len(parts) > 1:
                                    genotoxicity_classification = parts[1].strip().upper()
                                    break

                lc50_classification = classify_lc50_toxicity(ld50_lc50_data["LC50"])
                result["lc50_classification"] = lc50_classification
                if genotoxicity_classification == "INDETERMINADO" and lc50_classification["concern_level"] == "High":
                    genotoxicity_classification = "PRECAUTORIO"
                    result["precautionary_note"] = f"Classification elevated to PRECAUTORIO due to very low LC50 value ({lc50_classification['lowest_lc50']} {lc50_classification['unit']})"
                result["analysis"] = analysis
                result["genotoxicity_classification"] = genotoxicity_classification
                output_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.json")
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                md_output_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.md")
                with open(md_output_path, 'w') as f:
                    f.write(f"# Análisis de Datos de Toxicidad para CAS {cas}\n\n")
                    f.write("## Datos LD50/LC50 Extraídos\n\n")
                    f.write("```json\n")
                    f.write(json.dumps(ld50_lc50_data, indent=2))
                    f.write("\n```\n\n")
                    if genotoxicity_mentions:
                        f.write("## Menciones de Genotoxicidad Encontradas\n\n")
                        f.write(f"Se encontraron {len(genotoxicity_mentions)} menciones relacionadas con genotoxicidad o carcinogenicidad.\n\n")
                        for i, mention in enumerate(genotoxicity_mentions[:5]):
                            f.write(f"**{i+1}. Término:** '{mention['term']}'\n")
                            f.write(f"**Contexto:** \"...{mention['context']}...\"\n\n")
                    f.write("## Análisis\n\n")
                    f.write(analysis)
                    f.write("\n\n---\n")
                    f.write(f"**Clasificación de genotoxicidad**: {genotoxicity_classification}\n")
                    if lc50_classification["concern_level"] != "Unknown":
                        f.write(f"\n**Nivel de preocupación por LC50**: {lc50_classification['concern_level']}\n")
                        f.write(f"**Toxicidad LC50**: {lc50_classification['severity']}\n")
                        f.write(f"**Justificación**: {lc50_classification['reasoning']}\n")
                    if "precautionary_note" in result:
                        f.write(f"\n**NOTA PRECAUTORIA**: {result['precautionary_note']}\n")
                    f.write("\n*Análisis generado por GPT basado en todos los datos disponibles, incluyendo menciones directas de genotoxicidad y valores de LD50/LC50*")
                logger.info(f"Análisis completo de toxicidad para CAS {cas} guardado en {output_path}")
            except Exception as e:
                logger.error(f"Error al generar análisis con OpenAI: {str(e)}")
                result["error"] = f"Error al generar análisis: {str(e)}"
                md_output_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_data_raw.md")
                with open(md_output_path, 'w') as f:
                    f.write(f"# Datos de Toxicidad para CAS {cas}\n\n")
                    f.write(toxicity_text)
                    if genotoxicity_mentions:
                        f.write("\n## Menciones de Genotoxicidad Encontradas\n\n")
                        for i, mention in enumerate(genotoxicity_mentions):
                            f.write(f"**{i+1}. Término:** '{mention['term']}'\n")
                            f.write(f"**Contexto:** \"...{mention['context']}...\"\n\n")
            return result
        except Exception as e:
            logger.error(f"Error en flujo de trabajo de toxicidad para CAS {cas}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    def process_compound(self, cas):
        try:
            cid = self.get_cid_from_cas(cas)
            if not cid:
                logger.warning(f"No se pudo obtener CID para CAS {cas}")
                return False
            logger.info(f"Procesando compuesto CAS {cas} (CID: {cid})")
            existing_chunks = self.collection.get(where={"cas": cas}, limit=1)
            if existing_chunks["ids"]:
                logger.info(f"CAS {cas} ya existe en ChromaDB")
                toxicity_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.json")
                safety_path = os.path.join(SAFETY_DIR, f"{cas}_safety_analysis.json")
                # MODIFICACIÓN 3: Leer clasificación de genotoxicidad desde JSON (o markdown si falla)
                genotoxicity_analysis_file = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.json")
                genotoxic_status = "INDETERMINADO"
                if os.path.exists(genotoxicity_analysis_file):
                    try:
                        with open(genotoxicity_analysis_file, 'r') as f:
                            toxicity_data = json.load(f)
                            if "genotoxicity_classification" in toxicity_data:
                                genotoxic_status = toxicity_data["genotoxicity_classification"]
                    except:
                        genotoxicity_analysis_file = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.md")
                        if os.path.exists(genotoxicity_analysis_file):
                            with open(genotoxicity_analysis_file, 'r') as f:
                                analysis_content = f.read()
                                if "**Clasificación de genotoxicidad**: POSITIVO" in analysis_content:
                                    genotoxic_status = "POSITIVO"
                                elif "**Clasificación de genotoxicidad**: NEGATIVO" in analysis_content:
                                    genotoxic_status = "NEGATIVO"
                                elif "**Clasificación de genotoxicidad**: PRECAUTORIO" in analysis_content:
                                    genotoxic_status = "PRECAUTORIO"
                                elif "CLASIFICACIÓN_GENOTOXICIDAD: POSITIVO" in analysis_content:
                                    genotoxic_status = "POSITIVO"
                                elif "CLASIFICACIÓN_GENOTOXICIDAD: NEGATIVO" in analysis_content:
                                    genotoxic_status = "NEGATIVO"
                                elif "CLASIFICACIÓN_GENOTOXICIDAD: PRECAUTORIO" in analysis_content:
                                    genotoxic_status = "PRECAUTORIO"
                if os.path.exists(toxicity_path) and os.path.exists(safety_path):
                    logger.info(f"Análisis para CAS {cas} ya existen")
                    return True
                results = self.collection.get(where={"cas": cas}, limit=1000)
                chunks = results["documents"]
                if not os.path.exists(toxicity_path):
                    logger.info(f"Generando análisis de toxicidad para CAS {cas}")
                    # Forzamos la regeneración del análisis de toxicidad
                    toxicity_result = self.process_toxicity_data_workflow_optimized(cas)
                if not os.path.exists(safety_path):
                    logger.info(f"Generando análisis de seguridad para CAS {cas}")
                    self.analyze_safety(cas, chunks)
                return True
            compound_data = self.get_compound_data(cid)
            document = self.create_document_for_compound(cas, cid, compound_data)
            json_path = os.path.join(self.output_dir, f"{cas}_document.json")
            with open(json_path, 'w') as f:
                json.dump(document, f, indent=2)
            chunks = self.split_text_into_chunks(document['text'])
            logger.info(f"Compuesto CAS {cas} dividido en {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                chunk_path = os.path.join(self.output_dir, "chunks", f"{cas}_chunk_{i}.txt")
                with open(chunk_path, 'w') as f:
                    f.write(chunk)
            for i, chunk in enumerate(chunks):
                chunk_id = f"cas-{cas}-chunk-{i}"
                try:
                    if len(chunk) > 32000:
                        logger.warning(f"Chunk {i} demasiado grande, truncando...")
                        chunk = chunk[:32000]
                    embedding = self.get_embedding(chunk)
                    embedding_path = os.path.join(self.output_dir, "embeddings", f"{cas}_chunk_{i}_embedding.npy")
                    np.save(embedding_path, np.array(embedding))
                    embedding_info_path = os.path.join(self.output_dir, "embeddings", f"{cas}_chunk_{i}_embedding_info.json")
                    with open(embedding_info_path, 'w') as f:
                        json.dump({
                            "cas": cas,
                            "cid": cid,
                            "chunk_id": i,
                            "embedding_file": embedding_path,
                            "embedding_length": len(embedding),
                            "embedding_sample": embedding[:5],
                        }, f, indent=2)
                    self.collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[{
                            "cas": cas,
                            "cid": str(cid),
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "name": document["name"],
                            "formula": document["formula"],
                            "molecular_weight": str(document["weight"]),
                            "type": "compound_chunk",
                            "embedding_file": embedding_path
                        }],
                        ids=[chunk_id]
                    )
                    logger.info(f"Chunk {i+1}/{len(chunks)} del CAS {cas} procesado y añadido a ChromaDB")
                except Exception as e:
                    logger.error(f"Error al procesar chunk {i} del CAS {cas}: {str(e)}")
            try:
                self.chroma_client.persist()
                logger.info(f"ChromaDB persistida después de procesar CAS {cas}")
            except Exception as e:
                logger.warning(f"No se pudo persistir después de procesar CAS {cas}: {str(e)}")
            logger.info(f"Generando análisis de toxicidad para CAS {cas}")
            toxicity_result = self.process_toxicity_data_workflow_optimized(cas)
            logger.info(f"Generando análisis de seguridad para CAS {cas}")
            safety_result = self.analyze_safety(cas, chunks)
            return toxicity_result["success"] and safety_result["success"]
        except Exception as e:
            logger.error(f"Error al procesar CAS {cas}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def load_cached_embedding(self, embedding_file):
        try:
            if os.path.exists(embedding_file):
                embedding = np.load(embedding_file)
                return embedding.tolist()
            return None
        except Exception as e:
            logger.error(f"Error al cargar embedding desde caché: {str(e)}")
            return None

    def process_cas_list(self, cas_list):
        processed_count = 0
        for cas in tqdm(cas_list, desc="Procesando compuestos"):
            if self.process_compound(cas):
                processed_count += 1
                if processed_count % 5 == 0:
                    try:
                        self.chroma_client.persist()
                        logger.info(f"ChromaDB persistida después de {processed_count} compuestos")
                    except Exception as e:
                        logger.warning(f"Error al persistir después de {processed_count} compuestos: {str(e)}")
        try:
            self.chroma_client.persist()
            logger.info(f"ChromaDB persistida al finalizar {processed_count} compuestos")
        except Exception as e:
            logger.warning(f"Error al persistir al finalizar: {str(e)}")
        return processed_count

    def process_csv_file(self, csv_file, cas_column, delimiter=","):
        logger.info(f"=== INICIANDO PROCESAMIENTO DE CSV {csv_file} ===")
        analyzer = self
        genotoxic_compounds = []
        try:
            df = pd.read_csv(csv_file, delimiter=delimiter)
            print(f"Columnas disponibles en {csv_file}: {df.columns.tolist()}")
            if 'CAS_' in df.columns:
                df_valid = df[df['CAS_'] != '0-00-0'].copy()
                print(f"Se encontraron {len(df_valid)} compuestos con CAS válidos.")
                print("\nPrimeros 5 compuestos a procesar:")
                for i, (_, row) in enumerate(df_valid.head().iterrows()):
                    print(f"{i+1}. CAS: {row['CAS_']}, Nombre: {row.get('CompoundName', 'Sin nombre')}")
                
                process_all = input("\n¿Procesar todos los compuestos? (s/n): ").lower()
                if process_all == "s":
                    cas_list = df_valid['CAS_'].tolist()
                else:
                    try:
                        limit = int(input("Número de compuestos a procesar: "))
                        cas_list = df_valid['CAS_'].head(limit).tolist()
                    except ValueError:
                        print("Valor no válido. Procesando los primeros 10 compuestos.")
                        cas_list = df_valid['CAS_'].head(10).tolist()
                
                total_cas = len(cas_list)
                print(f"\nIniciando procesamiento de {total_cas} compuestos...")
                summary_data = []
                
                for i, cas in enumerate(cas_list):
                    print(f"\nProcesando compuesto {i+1}/{total_cas}: CAS {cas}")
                    compound_name = df[df['CAS_'] == cas]['CompoundName'].values[0] if cas in df['CAS_'].values else "Compuesto desconocido"
                    print(f"Nombre: {compound_name}")
                    
                    cid = analyzer.get_cid_from_cas(cas)
                    if not cid:
                        print(f"  ❌ No se pudo obtener CID para {cas}")
                        summary_data.append({
                            "CAS": cas,
                            "Nombre": compound_name,
                            "Procesado": "No",
                            "Error": "No se encontró CID",
                            "LD50_Count": 0,
                            "LC50_Count": 0,
                            "Genotóxico": "Desconocido"
                        })
                        continue
                    
                    try:
                        print(f"  → Extrayendo datos LD50/LC50 para CAS {cas} (CID: {cid})")
                        ld50_lc50_data = extract_ld50_lc50_direct(cas, cid)
                        
                        result_file = os.path.join("resultados_af016", f"{cas}_toxicity_data.json")
                        with open(result_file, 'w') as f:
                            json.dump(ld50_lc50_data, f, indent=2)
                        
                        summary_file = os.path.join("resultados_af016", f"{cas}_summary.txt")
                        with open(summary_file, 'w') as f:
                            f.write(f"RESUMEN DE TOXICIDAD PARA {compound_name} (CAS: {cas})\n")
                            f.write(f"CID PubChem: {cid}\n\n")
                            if ld50_lc50_data["LD50"]:
                                f.write("DATOS LD50 (Dosis Letal 50%):\n")
                                for item in ld50_lc50_data["LD50"]:
                                    f.write(f"- {item.get('valor')} {item.get('unidad')}")
                                    if item.get('especie'):
                                        f.write(f", {item.get('especie')}")
                                    if item.get('vía'):
                                        f.write(f", vía {item.get('vía')}")
                                    f.write("\n")
                                f.write("\n")
                            else:
                                f.write("No se encontraron datos LD50\n\n")
                            if ld50_lc50_data["LC50"]:
                                f.write("DATOS LC50 (Concentración Letal 50%):\n")
                                for item in ld50_lc50_data["LC50"]:
                                    f.write(f"- {item.get('valor')} {item.get('unidad')}")
                                    if item.get('especie'):
                                        f.write(f", {item.get('especie')}")
                                    if item.get('tiempo'):
                                        f.write(f", {item.get('tiempo')}")
                                    if item.get('via'):
                                        f.write(f", vía {item.get('via')}")
                                    f.write("\n")
                            else:
                                f.write("No se encontraron datos LC50\n")
                        
                        if ld50_lc50_data["LD50"] or ld50_lc50_data["LC50"]:
                            print(f"  → Realizando análisis completo de toxicidad...")
                            toxicity_result = analyzer.process_toxicity_data_workflow_optimized(cas)
                            
                            if toxicity_result.get("analysis"):
                                print("\n=== ANÁLISIS DE TOXICIDAD ===")
                                print(toxicity_result["analysis"])
                                print("===========================\n")
                                
                                # MODIFICACIÓN 1: Verificar la clasificación real
                                genotoxicity_status = toxicity_result.get("genotoxicity_classification", "INDETERMINADO")
                                
                                if genotoxicity_status == "POSITIVO":
                                    print("⚠️ ALERTA: Compuesto genotóxico/cancerígeno confirmado ⚠️")
                                    genotoxic_compounds.append((cas, compound_name))
                                elif genotoxicity_status == "NEGATIVO":
                                    print("✅ No se ha detectado genotoxicidad/carcinogenicidad")
                                elif genotoxicity_status == "PRECAUTORIO":
                                    print("🔶 PRECAUCIÓN: Posible riesgo genotóxico basado en valores LC50 muy bajos")
                                    genotoxic_compounds.append((cas, compound_name))
                                else:
                                    print("ℹ️ Datos insuficientes para evaluar genotoxicidad/carcinogenicidad")
                            else:
                                print("  ℹ️ No se pudo generar un análisis detallado.")
                        
                        print(f"  ✅ Datos guardados en {result_file} y {summary_file}")
                        
                        # MODIFICACIÓN 3: Leer la clasificación desde archivos JSON o markdown
                        toxicity_analysis_file = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.json")
                        genotoxic_status = "INDETERMINADO"
                        if os.path.exists(toxicity_analysis_file):
                            try:
                                with open(toxicity_analysis_file, 'r') as f:
                                    toxicity_data = json.load(f)
                                    if "genotoxicity_classification" in toxicity_data:
                                        genotoxic_status = toxicity_data["genotoxicity_classification"]
                            except:
                                toxicity_analysis_file = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.md")
                                if os.path.exists(toxicity_analysis_file):
                                    with open(toxicity_analysis_file, 'r') as f:
                                        analysis_content = f.read()
                                        if "**Clasificación de genotoxicidad**: POSITIVO" in analysis_content:
                                            genotoxic_status = "POSITIVO"
                                        elif "**Clasificación de genotoxicidad**: NEGATIVO" in analysis_content:
                                            genotoxic_status = "NEGATIVO"
                                        elif "**Clasificación de genotoxicidad**: PRECAUTORIO" in analysis_content:
                                            genotoxic_status = "PRECAUTORIO"
                                        elif "CLASIFICACIÓN_GENOTOXICIDAD: POSITIVO" in analysis_content:
                                            genotoxic_status = "POSITIVO"
                                        elif "CLASIFICACIÓN_GENOTOXICIDAD: NEGATIVO" in analysis_content:
                                            genotoxic_status = "NEGATIVO"
                                        elif "CLASIFICACIÓN_GENOTOXICIDAD: PRECAUTORIO" in analysis_content:
                                            genotoxic_status = "PRECAUTORIO"
                        summary_data.append({
                            "CAS": cas,
                            "Nombre": compound_name,
                            "Procesado": "Sí",
                            "Error": "Ninguno",
                            "LD50_Count": len(ld50_lc50_data["LD50"]),
                            "LC50_Count": len(ld50_lc50_data["LC50"]),
                            "Genotóxico": toxicity_result.get("genotoxicity_classification", genotoxic_status)
                        })
                    except Exception as e:
                        print(f"  ❌ Error procesando {cas}: {str(e)}")
                        summary_data.append({
                            "CAS": cas,
                            "Nombre": compound_name,
                            "Procesado": "Error",
                            "Error": str(e)[:100],
                            "LD50_Count": 0,
                            "LC50_Count": 0,
                            "Genotóxico": "Desconocido"
                        })
                    
                    if (i + 1) % 5 == 0 or (i + 1) == total_cas:
                        df_summary = pd.DataFrame(summary_data)
                        summary_csv = os.path.join("resultados_af016", "resumen_procesamiento.csv")
                        df_summary.to_csv(summary_csv, index=False)
                        print(f"\nResumen parcial guardado en {summary_csv}")
                
                print(f"\n=== PROCESAMIENTO COMPLETADO ===")
                print(f"Total compuestos procesados: {len(summary_data)}/{total_cas}")
                print(f"Resultados guardados en: {os.path.abspath('resultados_af016')}")
                
                df_summary = pd.DataFrame(summary_data)
                print("\nEstadísticas:")
                print(f"Compuestos con datos LD50: {len(df_summary[df_summary['LD50_Count'] > 0])}")
                print(f"Compuestos con datos LC50: {len(df_summary[df_summary['LC50_Count'] > 0])}")
                
                # MODIFICACIÓN 4: Actualizar el recuento final de clasificaciones
                genotoxic_count = len([s for s in df_summary['Genotóxico'] if s == "POSITIVO"])
                precautionary_count = len([s for s in df_summary['Genotóxico'] if s == "PRECAUTORIO"])
                negative_count = len([s for s in df_summary['Genotóxico'] if s == "NEGATIVO"])
                indeterminate_count = len([s for s in df_summary['Genotóxico'] if s not in ["POSITIVO", "NEGATIVO", "PRECAUTORIO"]])
                print(f"\n=== RESUMEN DE GENOTOXICIDAD ===")
                print(f"Compuestos genotóxicos confirmados: {genotoxic_count}/{len(df_summary)}")
                print(f"Compuestos con precaución genotóxica: {precautionary_count}/{len(df_summary)}")
                print(f"Compuestos sin potencial genotóxico: {negative_count}/{len(df_summary)}")
                print(f"Compuestos con datos insuficientes: {indeterminate_count}/{len(df_summary)}")
                
                if genotoxic_compounds:
                    print("\nCompuestos genotóxicos/cancerígenos confirmados:")
                    for idx, (cas_val, name) in enumerate(genotoxic_compounds, 1):
                        print(f"{idx}. {name} (CAS: {cas_val})")
                
                genotoxicity_report = os.path.join("resultados_af016", "informe_genotoxicidad.md")
                with open(genotoxicity_report, 'w') as f:
                    f.write("# Informe de Genotoxicidad\n\n")
                    f.write(f"Fecha: {time.strftime('%Y-%m-%d')}\n")
                    f.write(f"Total compuestos analizados: {len(df_summary)}\n")
                    f.write(f"Compuestos genotóxicos confirmados: {genotoxic_count}\n")
                    f.write(f"Compuestos con precaución genotóxica: {precautionary_count}\n")
                    f.write(f"Compuestos sin potencial genotóxico: {negative_count}\n")
                    f.write(f"Compuestos con datos insuficientes: {indeterminate_count}\n\n")
                    
                    if genotoxic_compounds:
                        f.write("## Compuestos Genotóxicos Confirmados\n\n")
                        for idx, (cas_val, name) in enumerate(genotoxic_compounds, 1):
                            f.write(f"### {idx}. {name} (CAS: {cas_val})\n\n")
                            toxicity_file = os.path.join(TOXICITY_DIR, f"{cas_val}_toxicity_analysis.md")
                            if os.path.exists(toxicity_file):
                                with open(toxicity_file, 'r') as analysis_file:
                                    content = analysis_file.read()
                                    f.write(content)
                                    f.write("\n\n---\n\n")
                    else:
                        f.write("No se identificaron compuestos genotóxicos confirmados en esta muestra.\n")
                
                print(f"\nInforme de genotoxicidad generado: {genotoxicity_report}")
            
            else:
                print(f"Error: Columna CAS_ no encontrada en el archivo CSV.")
        except Exception as e:
            print(f"Error al procesar el archivo CSV: {str(e)}")
            import traceback
            traceback.print_exc()

    def interactive_search(self):
        print("\n=== BÚSQUEDA INTERACTIVA DE COMPUESTOS ===\n")
        print("Escribe 'salir' para terminar.")
        while True:
            query = input("\nIntroduce tu consulta de búsqueda: ").strip()
            if query.lower() in ['salir', 'exit', 'quit']:
                break
            n_results = input("Número de resultados (por defecto 5): ").strip()
            try:
                n_results = int(n_results) if n_results else 5
            except:
                n_results = 5
            results = self.search_compounds(query, n_results=n_results)
            print(f"\nEncontrados {results['total_results']} resultados para '{query}':")
            for i, result in enumerate(results["results"]):
                print(f"\nResultado {i+1}: {result['metadata'].get('name', 'Desconocido')} (CAS: {result['metadata'].get('cas', 'N/A')})")
                print(f"Similitud: {1 - result['distance']:.2%}")
                print(f"Extracto: {result['document'][:200]}...")

    def analyze_toxicity_data_for_compound(self, cas="71-43-2"):
        print(f"=== INICIANDO ANÁLISIS COMPLETO DE TOXICIDAD PARA COMPUESTO CAS {cas} ===")
        result = self.process_toxicity_data_workflow_optimized(cas)
        if result["success"]:
            print(f"\n=== ANÁLISIS COMPLETO DE TOXICIDAD PARA CAS {cas} ===\n")
            print("DATOS EXTRAÍDOS DE PUBCHEM:")
            print(json.dumps(result["raw_data"], indent=2))
            print("\nANÁLISIS BASADO EN LOS DATOS:")
            print(result["analysis"])
            output_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.md")
            print(f"\nResultados guardados en: {output_path}")
        else:
            print(f"\nError en el análisis: {result.get('error', 'Desconocido')}")
        return result

def process_single_compound(cas="71-43-2"):
    logger.info(f"=== INICIANDO PROCESAMIENTO DE COMPUESTO {cas} ===")
    analyzer = ChemicalSafetyAnalyzer()
    cid = analyzer.get_cid_from_cas(cas)
    if not cid:
        logger.error(f"No se pudo obtener CID para el CAS {cas}")
        return False
    ld50_lc50_data = extract_ld50_lc50_direct(cas, cid)
    print("\n=== DATOS DE LD50/LC50 EXTRAÍDOS ===\n")
    print(json.dumps(ld50_lc50_data, indent=2))
    success = analyzer.process_compound(cas)
    if success:
        logger.info(f"Compuesto CAS {cas} procesado correctamente")
        toxicity_analysis = analyzer.get_toxicity_analysis(cas)
        if toxicity_analysis and toxicity_analysis.get("success"):
            print("\n=== ANÁLISIS DE TOXICIDAD ===\n")
            print(toxicity_analysis.get("toxicity_analysis", "No disponible"))
        safety_analysis = analyzer.get_safety_analysis(cas)
        if safety_analysis and safety_analysis.get("success"):
            print("\n=== ANÁLISIS DE SEGURIDAD ===\n")
            print(safety_analysis.get("safety_analysis", "No disponible"))
        print("\n=== BÚSQUEDA SEMÁNTICA DE EJEMPLO ===\n")
        search_results = analyzer.search_compounds("carcinogenic properties", n_results=3)
        if search_results["total_results"] > 0:
            print(f"Resultados para 'carcinogenic properties': {search_results['total_results']}")
            for i, result in enumerate(search_results["results"]):
                print(f"Resultado {i+1}: {result['id']} (Distancia: {result['distance']})")
                print(f"Extracto: {result['document'][:150]}...\n")
    else:
        logger.error(f"Error al procesar compuesto CAS {cas}")

def extract_ld50_lc50_for_compound(cas="71-43-2"):
    print(f"=== EXTRAYENDO DATOS LD50/LC50 PARA COMPUESTO CAS {cas} ===")
    analyzer = ChemicalSafetyAnalyzer()
    cid = analyzer.get_cid_from_cas(cas)
    if not cid:
        print(f"No se pudo obtener CID para el CAS {cas}")
        return None
    ld50_lc50_data = extract_ld50_lc50_direct(cas, cid)
    print(f"\n=== DATOS DE LD50/LC50 PARA CAS {cas} ===\n")
    print(json.dumps(ld50_lc50_data, indent=2))
    if ld50_lc50_data["LD50"]:
        print("\nValores LD50 (Dosis Letal 50%):")
        for i, item in enumerate(ld50_lc50_data["LD50"], 1):
            print(f"{i}. Valor: {item.get('valor')} {item.get('unidad')}")
            print(f"   Especie: {item.get('especie', 'No especificada')}")
            print(f"   Vía: {item.get('vía', 'No especificada')}")
            if 'source' in item:
                print(f"   Fuente: {item.get('source', 'No especificada')}")
            print("")
    if ld50_lc50_data["LC50"]:
        print("\nValores LC50 (Concentración Letal 50%):")
        for i, item in enumerate(ld50_lc50_data["LC50"], 1):
            print(f"{i}. Valor: {item.get('valor')} {item.get('unidad')}")
            print(f"   Especie: {item.get('especie', 'No especificada')}")
            print(f"   Tiempo: {item.get('tiempo', 'No especificado')}")
            print(f"   Vía: {item.get('via', 'No especificada')}")
            if 'source' in item:
                print(f"   Fuente: {item.get('source', 'No especificada')}")
            print("")
    return ld50_lc50_data

def analyze_toxicity_data_for_compound(cas="71-43-2"):
    print(f"=== INICIANDO ANÁLISIS COMPLETO DE TOXICIDAD PARA COMPUESTO CAS {cas} ===")
    analyzer = ChemicalSafetyAnalyzer()
    result = analyzer.process_toxicity_data_workflow_optimized(cas)
    if result["success"]:
        print(f"\n=== ANÁLISIS COMPLETO DE TOXICIDAD PARA CAS {cas} ===\n")
        print("DATOS EXTRAÍDOS DE PUBCHEM:")
        print(json.dumps(result["raw_data"], indent=2))
        print("\nANÁLISIS BASADO EN LOS DATOS:")
        print(result["analysis"])
        output_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.md")
        print(f"\nResultados guardados en: {output_path}")
    else:
        print(f"\nError en el análisis: {result.get('error', 'Desconocido')}")
    return result

def main():
    print("=== INICIANDO ANÁLISIS DE TOXICIDAD PARA COMPUESTOS EN AF016.csv ===")
    results_dir = "resultados_af016"
    os.makedirs(results_dir, exist_ok=True)
    analyzer = ChemicalSafetyAnalyzer()
    csv_file = "AF016.csv"
    analyzer.process_csv_file(csv_file, "CAS_")
    # descomentar las siguientes líneas para probar otras funciones:
    # process_single_compound("71-43-2")
    # analyzer.interactive_search()
    # analyze_toxicity_data_for_compound("71-43-2")

if __name__ == "__main__":
    main()
