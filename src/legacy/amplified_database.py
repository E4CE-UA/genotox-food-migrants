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
import subprocess
import argparse

# Configuración del logger
logging.basicConfig(
    level=logging.INFO,  # <--- Si deseas ver logs debug, cámbialo a logging.DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chemical_safety_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============= CONFIGURACIÓN =============
# Configuración OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Tamaño máximo de chunk (en caracteres, aproximado)
MAX_CHUNK_SIZE = 4000  # Aproximadamente 4000 caracteres ~ 1000 tokens

# Configuración de la base de datos ChromaDB
CHROMADB_COLLECTION = "chemicals_collection"
CHROMA_PERSIST_DIR = "./chroma_db"  # Directorio para persistencia de ChromaDB

# Directorio de salida
OUTPUT_DIR = "chemicals_database"

# Subdirectorios para categorías específicas
TOXICITY_DIR = os.path.join(OUTPUT_DIR, "toxicity_analysis")
SAFETY_DIR = os.path.join(OUTPUT_DIR, "safety_analysis")

# Tiempo de espera entre solicitudes a la API de PubChem (para evitar sobrecarga)
API_SLEEP_TIME = 1

# Modelo GPT para análisis de toxicidad y seguridad
GPT_MODEL = "gpt-4o"
# =========================================

# Cliente OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Crear directorios necesarios
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "json"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "chunks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "embeddings"), exist_ok=True)
os.makedirs(TOXICITY_DIR, exist_ok=True)
os.makedirs(SAFETY_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)  # Asegurar que el directorio de persistencia exista

class ChemicalSafetyAnalyzer:
    def __init__(self, output_dir=OUTPUT_DIR, collection_name=CHROMADB_COLLECTION, persist_dir=CHROMA_PERSIST_DIR):
        """
        Inicializa el analizador de seguridad química
        
        Args:
            output_dir (str): Directorio para almacenar datos
            collection_name (str): Nombre de la colección en ChromaDB
            persist_dir (str): Directorio para la persistencia de ChromaDB
        """
        self.output_dir = output_dir
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self.cas_to_cid_map = {}
        
        # Inicializar cliente ChromaDB con persistencia
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.persist_dir
            )
            logger.info(f"Cliente ChromaDB inicializado con persistencia en {self.persist_dir}")
        except Exception as e:
            logger.error(f"Error al inicializar ChromaDB: {str(e)}")
            raise
        
        # Crear/obtener colección con configuración para guardar embeddings
        try:
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Colección ChromaDB '{collection_name}' inicializada")
        except Exception as e:
            logger.error(f"Error al inicializar colección ChromaDB: {str(e)}")
            raise
        
        # Crear directorios necesarios para CompTox
        comptox_dir = os.path.join(self.output_dir, "comptox")
        os.makedirs(comptox_dir, exist_ok=True)
    
    def get_embedding(self, text, model="text-embedding-3-small"):
        """
        Obtiene el embedding de un texto usando la API de OpenAI
        
        Args:
            text (str): Texto para generar embedding
            model (str): Modelo de OpenAI a utilizar
            
        Returns:
            list: Vector de embedding
        """
        if not text or len(text.strip()) == 0:
            return [0.0] * 1536
        text = text.replace("\n", " ")
        try:
            response = openai_client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error al obtener embedding: {str(e)}")
            return [0.0] * 1536
    
    def get_cid_from_cas(self, cas):
        """
        Obtiene el CID de PubChem a partir de un número CAS
        
        Args:
            cas (str): Número CAS
            
        Returns:
            int or None: CID de PubChem o None si no se encuentra
        """
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
        """
        Obtiene datos completos de un compuesto a partir de su CID
        
        Args:
            cid (int): CID de PubChem
            
        Returns:
            dict: Datos del compuesto
        """
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
        """
        Extrae texto de una sección de datos de PubChem
        
        Args:
            section_data (dict): Datos de la sección
            
        Returns:
            str: Texto extraído
        """
        texts = []
        def extract_from_section(section):
            if isinstance(section, dict):
                if 'Value' in section and 'StringWithMarkup' in section['Value']:
                    for item in section['Value']['StringWithMarkup']:
                        if 'String' in item:
                            texts.append(item['String'])
                if 'Section' in section:
                    for subsection in section['Section']:
                        extract_from_section(subsection)
                if 'Information' in section:
                    for info in section['Information']:
                        extract_from_section(info)
            elif isinstance(section, list):
                for item in section:
                    extract_from_section(item)
        if 'Record' in section_data and 'Section' in section_data['Record']:
            extract_from_section(section_data['Record'])
        return "\n".join(texts)
    
    def split_text_into_chunks(self, text, max_chunk_size=MAX_CHUNK_SIZE):
        """
        Divide el texto en chunks más pequeños
        
        Args:
            text (str): Texto a dividir
            max_chunk_size (int): Tamaño máximo de cada chunk en caracteres
            
        Returns:
            list: Lista de chunks
        """
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        for paragraph in paragraphs:
            if len(paragraph) > max_chunk_size:
                sentences = paragraph.replace(". ", ".\n").split("\n")
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= max_chunk_size:
                        if current_chunk:
                            current_chunk += "\n\n" + sentence
                        else:
                            current_chunk = sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                if len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                    if current_chunk:
                        current_chunk += "\n\n" + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    chunks.append(current_chunk)
                    current_chunk = paragraph
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    
    def create_document_for_compound(self, cas, cid, compound_data):
        """
        Crea un documento estructurado para el compuesto
        
        Args:
            cas (str): Número CAS
            cid (int): CID de PubChem
            compound_data (dict): Datos del compuesto
            
        Returns:
            dict: Documento estructurado
        """
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
        full_text += f"Name: {document['name']}\n"
        full_text += f"Formula: {document['formula']}\n"
        full_text += f"Molecular Weight: {document['weight']}\n"
        full_text += f"XLogP: {document['xlogp']}\n"
        full_text += f"SMILES: {document['smiles']}\n\n"
        for key, text in text_sections.items():
            title = key.replace('_', ' ').title()
            full_text += f"--- {title} ---\n{text}\n\n"
        document['text'] = full_text
        document['sections'] = text_sections
        return document
    
    def extract_toxicity_sections(self, chunks):
        """
        Extrae secciones específicas de toxicidad de los chunks
        
        Args:
            chunks (list): Lista de chunks de texto
            
        Returns:
            str: Texto combinado de las secciones relevantes
        """
        relevant_text = []
        toxicity_patterns = [
            r"(?i)--- Toxicity ---.*?(?=---|\Z)",
            r"(?i)--- Safety and Hazards ---.*?(?=---|\Z)",
            r"(?i)--- GHS Classification ---.*?(?=---|\Z)",
            r"(?i)Toxicity[:\s].*?(?=\n\n|\Z)",
            r"(?i)Hazard.*?(?=\n\n|\Z)",
            r"(?i)Carcinogen.*?(?=\n\n|\Z)",
            r"(?i)Mutagen.*?(?=\n\n|\Z)",
            r"(?i)toxic effect.*?(?=\n\n|\Z)",
            r"(?i)health hazard.*?(?=\n\n|\Z)",
            r"(?i)risk phrase.*?(?=\n\n|\Z)",
            r"(?i)safety phrase.*?(?=\n\n|\Z)",
            r"(?i)LD50.*?(?=\n\n|\Z)",
            r"(?i)LC50.*?(?=\n\n|\Z)"
        ]
        for chunk in chunks:
            for pattern in toxicity_patterns:
                matches = re.findall(pattern, chunk, re.DOTALL)
                for match in matches:
                    if match and len(match.strip()) > 10:
                        relevant_text.append(match.strip())
        if not relevant_text:
            return "\n\n".join(chunks)
        return "\n\n".join(relevant_text)
    
    def analyze_toxicity(self, cas, chunks):
        """
        Analiza la toxicidad de un compuesto usando OpenAI GPT, incorporando datos de CompTox
        
        Args:
            cas (str): Número CAS del compuesto
            chunks (list): Lista de chunks de texto con información del compuesto
            
        Returns:
            dict: Resultado del análisis
        """
        try:
            logger.info(f"Analizando toxicidad del compuesto CAS {cas} con {GPT_MODEL}")
            toxicity_text = self.extract_toxicity_sections(chunks)
            # Intentar cargar datos de CompTox
            comptox_dir = os.path.join(self.output_dir, "comptox")
            comptox_json_path = os.path.join(comptox_dir, f"{cas}_comptox.json")
            comptox_data = None
            if os.path.exists(comptox_json_path):
                try:
                    with open(comptox_json_path, 'r') as f:
                        comptox_data = json.load(f)
                    logger.info(f"Datos de CompTox cargados para análisis de toxicidad de CAS {cas}")
                except Exception as e:
                    logger.warning(f"Error al cargar datos de CompTox para análisis: {str(e)}")
            comptox_section = ""
            if comptox_data:
                comptox_section = "\n--- DATOS DE COMPTOX EPA ---\n"
                toxicity_keys = [
                    "DTXSID", "Toxicity", "LogP", "HalfLife", "WaterSolubility", 
                    "Carcinogenicity", "Mutagenicity", "Genotoxicity", "EndocrineActivity",
                    "AcuteToxicity", "ChronicToxicity", "DevelopmentalToxicity",
                    "ReproductiveToxicity", "NeurologicalEffects", "ImmunotoxicityEffects",
                    "LOEL", "NOEL", "LD50", "LC50", "EC50", "TD50", "IDLH"
                ]
                for key in comptox_data:
                    if key in toxicity_keys or "toxic" in key.lower() or "hazard" in key.lower():
                        if comptox_data[key] and str(comptox_data[key]).strip().lower() != 'nan':
                            comptox_section += f"{key}: {comptox_data[key]}\n"
                if comptox_section == "\n--- DATOS DE COMPTOX EPA ---\n":
                    for key, value in comptox_data.items():
                        if value and str(value).strip().lower() != 'nan':
                            comptox_section += f"{key}: {value}\n"
            combined_text = toxicity_text + "\n\n" + comptox_section if comptox_section else toxicity_text
            if len(combined_text) > 20000:
                logger.warning(f"Texto demasiado largo ({len(combined_text)} caracteres), recortando...")
                if len(comptox_section) < 5000:
                    max_pubchem = 20000 - len(comptox_section)
                    toxicity_text = toxicity_text[:max_pubchem] + "..."
                    combined_text = toxicity_text + "\n\n" + comptox_section
                else:
                    combined_text = combined_text[:20000] + "..."
            prompt = f"""
            Analiza la siguiente información sobre un compuesto químico con CAS {cas} y evalúa su toxicidad y peligrosidad:

            {combined_text}

            Basándote en esta información, proporciona un análisis detallado y estructurado que incluya:
            1. Resumen de la toxicidad del compuesto (párrafo breve con lo más destacado)
            2. Clasificación de peligrosidad (baja, media, alta, muy alta) con justificación
            3. Principales riesgos para la salud humana, ordenados por severidad
            4. Posibles efectos a largo plazo, incluyendo cancerígenos, mutagénicos o teratogénicos si aplica
            5. Límites de exposición recomendados (si están disponibles)
            6. Medidas de seguridad y prevención recomendadas
            7. Conclusión y recomendación final sobre el manejo del compuesto

            Considera ESPECIALMENTE los datos de CompTox EPA si están disponibles, ya que contienen información cuantitativa importante sobre toxicidad.
            Si alguna información no está disponible en el texto proporcionado, indícalo claramente.
            Usa un formato bien estructurado con títulos claros para cada sección.
            """
            logger.info(f"Consultando a {GPT_MODEL} para análisis de toxicidad del CAS {cas}")
            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "Eres un experto en toxicología y seguridad química que proporciona análisis precisos, profesionales y bien estructurados de la toxicidad de compuestos químicos. Tus análisis son concisos pero completos, y destacan los puntos más importantes para la seguridad. Integras datos de diversas fuentes incluyendo PubChem y CompTox EPA para ofrecer una evaluación completa."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            analysis = response.choices[0].message.content.strip()
            result = {
                "cas": cas,
                "success": True,
                "toxicity_analysis": analysis,
                "analyzed_text_length": len(combined_text),
                "chunks_analyzed": len(chunks),
                "model_used": GPT_MODEL,
                "comptox_data_included": comptox_data is not None
            }
            output_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            md_output_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.md")
            with open(md_output_path, 'w') as f:
                f.write(f"# Análisis de Toxicidad para Compuesto CAS {cas}\n\n")
                f.write(analysis)
                f.write("\n\n---\n*Análisis generado por GPT basado en datos de PubChem y CompTox EPA*")
            logger.info(f"Análisis de toxicidad para CAS {cas} completado y guardado en {output_path}")
            return result
        except Exception as e:
            logger.error(f"Error al analizar toxicidad del CAS {cas}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "cas": cas,
                "success": False,
                "error": str(e),
                "toxicity_analysis": None
            }
    
    def extract_safety_sections(self, chunks):
        """
        Extrae secciones específicas de seguridad de los chunks
        
        Args:
            chunks (list): Lista de chunks de texto
            
        Returns:
            str: Texto combinado de las secciones relevantes
        """
        relevant_text = []
        safety_patterns = [
            r"(?i)--- Safety and Hazards ---.*?(?=---|\Z)",
            r"(?i)--- GHS Classification ---.*?(?=---|\Z)",
            r"(?i)Safety.*?(?=\n\n|\Z)",
            r"(?i)Hazard.*?(?=\n\n|\Z)",
            r"(?i)Protection.*?(?=\n\n|\Z)",
            r"(?i)PPE.*?(?=\n\n|\Z)",
            r"(?i)First Aid.*?(?=\n\n|\Z)",
            r"(?i)Emergency.*?(?=\n\n|\Z)",
            r"(?i)Handle.*?(?=\n\n|\Z)",
            r"(?i)Store.*?(?=\n\n|\Z)",
            r"(?i)Firefighting.*?(?=\n\n|\Z)",
            r"(?i)Precaution.*?(?=\n\n|\Z)",
            r"(?i)Spill.*?(?=\n\n|\Z)"
        ]
        for chunk in chunks:
            for pattern in safety_patterns:
                matches = re.findall(pattern, chunk, re.DOTALL)
                for match in matches:
                    if match and len(match.strip()) > 10:
                        relevant_text.append(match.strip())
        if not relevant_text:
            return "\n\n".join(chunks)
        return "\n\n".join(relevant_text)
    
    def analyze_safety(self, cas, chunks):
        """
        Analiza la seguridad y medidas de protección de un compuesto usando OpenAI GPT
        
        Args:
            cas (str): Número CAS del compuesto
            chunks (list): Lista de chunks de texto con información del compuesto
            
        Returns:
            dict: Resultado del análisis
        """
        try:
            logger.info(f"Analizando seguridad del compuesto CAS {cas} con {GPT_MODEL}")
            safety_text = self.extract_safety_sections(chunks)
            if len(safety_text) > 20000:
                logger.warning(f"Texto demasiado largo ({len(safety_text)} caracteres), recortando...")
                safety_text = safety_text[:20000] + "..."
            prompt = f"""
            Analiza la siguiente información sobre seguridad y manejo para un compuesto químico con CAS {cas}:

            {safety_text}

            Basándote en esta información, proporciona una guía de seguridad detallada y estructurada que incluya:
            1. Resumen de los principales peligros (inflamabilidad, reactividad, riesgos para la salud, etc.)
            2. Equipo de protección personal (EPP) recomendado para su manejo
            3. Condiciones de almacenamiento adecuadas
            4. Procedimientos de manejo seguro
            5. Medidas en caso de derrame o liberación accidental
            6. Incompatibilidades con otros materiales
            7. Procedimientos de primeros auxilios
            8. Medidas de extinción de incendios
            9. Consideraciones de eliminación de residuos
            10. Legislación y normativa aplicable (si se menciona)

            Si alguna información no está disponible en el texto proporcionado, indícalo claramente.
            Usa un formato bien estructurado con títulos claros para cada sección.
            """
            logger.info(f"Consultando a {GPT_MODEL} para análisis de seguridad del CAS {cas}")
            response = openai_client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": "Eres un especialista en seguridad química que proporciona guías prácticas, precisas y bien estructuradas para el manejo seguro de productos químicos. Tus guías destacan los aspectos más críticos para la seguridad de los trabajadores y el medio ambiente."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            analysis = response.choices[0].message.content.strip()
            result = {
                "cas": cas,
                "success": True,
                "safety_analysis": analysis,
                "analyzed_text_length": len(safety_text),
                "chunks_analyzed": len(chunks),
                "model_used": GPT_MODEL
            }
            output_path = os.path.join(SAFETY_DIR, f"{cas}_safety_analysis.json")
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            md_output_path = os.path.join(SAFETY_DIR, f"{cas}_safety_analysis.md")
            with open(md_output_path, 'w') as f:
                f.write(f"# Guía de Seguridad para Compuesto CAS {cas}\n\n")
                f.write(analysis)
                f.write("\n\n---\n*Análisis generado por GPT basado en datos de PubChem*")
            logger.info(f"Análisis de seguridad para CAS {cas} completado y guardado en {output_path}")
            return result
        except Exception as e:
            logger.error(f"Error al analizar seguridad del CAS {cas}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "cas": cas,
                "success": False,
                "error": str(e),
                "safety_analysis": None
            }
    
    def process_compound(self, cas):
        """
        Procesa un compuesto específico por su CAS, genera embeddings y realiza análisis
        
        Args:
            cas (str): Número CAS del compuesto
            
        Returns:
            bool: True si se procesó correctamente
        """
        try:
            cid = self.get_cid_from_cas(cas)
            if not cid:
                logger.warning(f"No se pudo obtener CID para CAS {cas}")
                return False
            logger.info(f"Procesando compuesto CAS {cas} (CID: {cid})")
            existing_chunks = self.collection.get(
                where={"cas": cas},
                limit=1
            )
            if existing_chunks["ids"]:
                logger.info(f"CAS {cas} ya existe en ChromaDB ({len(existing_chunks['ids'])} chunks)")
                toxicity_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.json")
                safety_path = os.path.join(SAFETY_DIR, f"{cas}_safety_analysis.json")
                if os.path.exists(toxicity_path) and os.path.exists(safety_path):
                    logger.info(f"Análisis de toxicidad y seguridad para CAS {cas} ya existen")
                    return True
                results = self.collection.get(
                    where={"cas": cas},
                    limit=1000
                )
                chunks = results["documents"]
                if not os.path.exists(toxicity_path):
                    logger.info(f"Generando análisis de toxicidad para CAS {cas}")
                    self.analyze_toxicity(cas, chunks)
                if not os.path.exists(safety_path):
                    logger.info(f"Generando análisis de seguridad para CAS {cas}")
                    self.analyze_safety(cas, chunks)
                return True
            compound_data = self.get_compound_data(cid)
            
            # NUEVO: Obtener datos de CompTox EPA (con debug)
            comptox_data = self.get_comptox_data_debug(cas)  # <--- Usamos la versión debug
            # Si prefieres no crear una versión aparte, simplemente cambia get_comptox_data
            # en tu código y llama get_comptox_data(cas) directamente.
            
            document = self.create_document_for_compound(cas, cid, compound_data)
            # NUEVO: Integrar datos de CompTox en el documento
            if comptox_data:
                comptox_text = "--- CompTox EPA Data ---\n"
                for key, value in comptox_data.items():
                    if value and str(value).strip() and str(value).lower() != 'nan':
                        comptox_text += f"{key}: {value}\n"
                document['text'] += "\n" + comptox_text
                document['sections']['comptox'] = comptox_text
                logger.info(f"Datos de CompTox integrados para CAS {cas}")
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
                        logger.warning(f"Chunk {i} demasiado grande ({len(chunk)} caracteres), truncando...")
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
                    metadata = {
                        "cas": cas,
                        "cid": str(cid),
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "name": document["name"],
                        "formula": document["formula"],
                        "molecular_weight": str(document["weight"]),
                        "type": "compound_chunk",
                        "embedding_file": embedding_path,
                        "has_comptox_data": comptox_data is not None
                    }
                    if comptox_data:
                        comptox_keys = ["DTXSID", "Toxicity", "LogP", "HalfLife", "WaterSolubility"]
                        for key in comptox_keys:
                            if key in comptox_data and comptox_data[key] and str(comptox_data[key]).lower() != 'nan':
                                metadata[f"comptox_{key}"] = str(comptox_data[key])
                    self.collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata],
                        ids=[chunk_id]
                    )
                    logger.info(f"Chunk {i+1}/{len(chunks)} del CAS {cas} procesado y añadido a ChromaDB")
                except Exception as e:
                    logger.error(f"Error al procesar chunk {i} del CAS {cas}: {str(e)}")
            try:
                self.chroma_client.persist()
                logger.info(f"Base de datos ChromaDB persistida después de procesar CAS {cas}")
            except Exception as e:
                logger.warning(f"No se pudo forzar persistencia para CAS {cas}: {str(e)}")
            logger.info(f"Generando análisis de toxicidad para CAS {cas}")
            toxicity_result = self.analyze_toxicity(cas, chunks)
            logger.info(f"Generando análisis de seguridad para CAS {cas}")
            safety_result = self.analyze_safety(cas, chunks)
            return toxicity_result["success"] and safety_result["success"]
        except Exception as e:
            logger.error(f"Error al procesar CAS {cas}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def get_comptox_data_debug(self, cas):
        """
        Versión con debug adicional que muestra las primeras líneas del CSV,
        el tamaño del archivo, etc.
        """
        try:
            logger.info(f"[DEBUG] Descargando datos de CompTox EPA para CAS {cas} con mayor detalle")
            comptox_dir = os.path.join(self.output_dir, "comptox")
            os.makedirs(comptox_dir, exist_ok=True)
            csv_path = os.path.join(comptox_dir, f"{cas}_comptox.csv")
            
            if os.path.exists(csv_path):
                logger.info(f"Usando datos de CompTox EPA previamente descargados para CAS {cas}")
            else:
                url = f"https://comptox.epa.gov/dashboard/dsstoxdb/results?search={cas}&download=csv"
                logger.info(f"Ejecutando curl para descargar datos de CompTox para CAS {cas}")
                
                try:
                    # 1) Intentar descarga con 'curl' vía subprocess
                    result = subprocess.run(
                        ["curl", "-s", url],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    # Guardar la respuesta en un archivo CSV
                    with open(csv_path, 'w') as f:
                        f.write(result.stdout)
                    
                    logger.info(f"Datos de CompTox descargados y guardados en {csv_path}")
                    
                    # Mostrar las primeras líneas del CSV para debug
                    csv_lines = result.stdout.splitlines()
                    logger.debug("Primeras líneas del CSV descargado (máx 10):")
                    for i, line in enumerate(csv_lines[:10]):
                        logger.debug(f"    {line}")
                        
                except subprocess.CalledProcessError as e:
                    logger.error(f"Error al ejecutar curl: {str(e)}")
                    
                    # 2) Si falla 'curl', intentar con requests como respaldo
                    try:
                        logger.info("[DEBUG] Intentando descarga con requests como respaldo")
                        response = requests.get(url)
                        
                        if response.status_code == 200:
                            with open(csv_path, 'w') as f:
                                f.write(response.text)
                            
                            logger.info(f"Datos de CompTox descargados con requests y guardados en {csv_path}")
                            
                            # (Opcional) Mostrar primeras líneas para debug
                            csv_lines = response.text.splitlines()
                            logger.debug("Primeras líneas del CSV descargado (máx 10):")
                            for i, line in enumerate(csv_lines[:10]):
                                logger.debug(f"    {line}")
                        else:
                            logger.error(f"Error en la solicitud HTTP: {response.status_code}")
                            return None
                            
                    except Exception as req_error:
                        logger.error(f"Error al descargar con requests: {str(req_error)}")
                        return None
            
            # Procesar el CSV descargado
            try:
                file_size = os.path.getsize(csv_path)
                logger.info(f"El archivo {csv_path} tiene un tamaño de {file_size} bytes")
                
                df = pd.read_csv(csv_path)
                
                if df.empty:
                    logger.warning(f"El CSV de CompTox para CAS {cas} está vacío o no contiene filas")
                    logger.debug(f"Columnas detectadas por pandas: {df.columns.tolist()}")
                    return None
                
                # Tomar el primer registro
                comptox_data = df.to_dict(orient='records')[0]
                
                json_path = os.path.join(comptox_dir, f"{cas}_comptox.json")
                with open(json_path, 'w') as f:
                    json.dump(comptox_data, f, indent=2)
                
                logger.info(f"Datos de CompTox procesados para CAS {cas}: {len(comptox_data)} campos")
                return comptox_data
            
            except Exception as parse_error:
                logger.error(f"Error al procesar CSV de CompTox con pandas: {str(parse_error)}")
                return None
        
        except Exception as e:
            logger.error(f"Error general al obtener datos de CompTox para CAS {cas}: {str(e)}")
            return None
    
    def load_cached_embedding(self, embedding_file):
        """
        Carga un embedding desde un archivo de caché si existe
        
        Args:
            embedding_file (str): Ruta al archivo de embedding
            
        Returns:
            list: Vector de embedding o None si no existe
        """
        try:
            if os.path.exists(embedding_file):
                embedding = np.load(embedding_file)
                return embedding.tolist()
            return None
        except Exception as e:
            logger.error(f"Error al cargar embedding desde caché: {str(e)}")
            return None
    
    def process_cas_list(self, cas_list):
        """
        Procesa una lista de números CAS
        
        Args:
            cas_list (list): Lista de números CAS
            
        Returns:
            int: Número de compuestos procesados
        """
        processed_count = 0
        for cas in tqdm(cas_list, desc="Procesando compuestos"):
            if self.process_compound(cas):
                processed_count += 1
                if processed_count % 5 == 0:
                    try:
                        self.chroma_client.persist()
                        logger.info(f"Base de datos ChromaDB persistida después de {processed_count} compuestos")
                    except Exception as e:
                        logger.warning(f"No se pudo forzar persistencia después de {processed_count} compuestos: {str(e)}")
        try:
            self.chroma_client.persist()
            logger.info(f"Base de datos ChromaDB persistida al finalizar el procesamiento de {processed_count} compuestos")
        except Exception as e:
            logger.warning(f"No se pudo forzar persistencia al finalizar: {str(e)}")
        return processed_count
    
    def process_csv_file(self, csv_file, cas_column, delimiter=","):
        """
        Procesa un archivo CSV con números CAS
        
        Args:
            csv_file (str): Ruta al archivo CSV
            cas_column (str): Nombre de la columna con números CAS
            delimiter (str): Delimitador del CSV
            
        Returns:
            int: Número de compuestos procesados
        """
        try:
            logger.info(f"Leyendo archivo CSV: {csv_file}")
            df = pd.read_csv(csv_file, delimiter=delimiter)
            if cas_column not in df.columns:
                logger.error(f"La columna '{cas_column}' no existe en el CSV. Columnas disponibles: {df.columns.tolist()}")
                return 0
            cas_list = df[cas_column].astype(str).tolist()
            cas_list = [cas.strip() for cas in cas_list if cas and cas.strip() and cas.strip().lower() != 'nan']
            logger.info(f"Extraídos {len(cas_list)} números CAS de {csv_file}")
            return self.process_cas_list(cas_list)
        except Exception as e:
            logger.error(f"Error al procesar archivo CSV: {str(e)}")
            return 0
    
    def search_compounds(self, query, n_results=5, cas_filter=None, comptox_filter=None):
        """
        Realiza una búsqueda semántica en la colección de compuestos con opción de filtrar por datos de CompTox
        
        Args:
            query (str): Consulta de texto
            n_results (int): Número de resultados a devolver
            cas_filter (str, optional): Filtrar resultados por CAS específico
            comptox_filter (dict, optional): Filtrar por propiedades de CompTox (ej: {"comptox_Toxicity": "high"})
            
        Returns:
            dict: Resultados de la búsqueda estructurados
        """
        try:
            query_embedding = self.get_embedding(query)
            search_params = {
                "query_embeddings": [query_embedding],
                "n_results": n_results
            }
            where_clause = {}
            if cas_filter:
                where_clause["cas"] = cas_filter
            if comptox_filter and isinstance(comptox_filter, dict):
                for key, value in comptox_filter.items():
                    where_clause[key] = value
            if where_clause:
                search_params["where"] = where_clause
            results = self.collection.query(**search_params)
            structured_results = []
            if results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    metadata = results["metadatas"][0][i]
                    comptox_metadata = {}
                    for key, value in metadata.items():
                        if key.startswith("comptox_"):
                            comptox_metadata[key] = value
                    structured_results.append({
                        "id": results["ids"][0][i],
                        "metadata": metadata,
                        "comptox_data": comptox_metadata if comptox_metadata else None,
                        "distance": results["distances"][0][i],
                        "document": results["documents"][0][i][:300] + "..." if len(results["documents"][0][i]) > 300 else results["documents"][0][i]
                    })
            return {
                "query": query,
                "total_results": len(structured_results),
                "comptox_filter_applied": comptox_filter is not None,
                "results": structured_results
            }
        except Exception as e:
            logger.error(f"Error en búsqueda semántica: {str(e)}")
            return {"query": query, "total_results": 0, "results": []}
    
    def get_compound_info(self, cas):
        """
        Recupera información completa de un compuesto a partir de todos sus chunks
        
        Args:
            cas (str): Número CAS del compuesto
            
        Returns:
            dict: Información completa del compuesto
        """
        try:
            json_path = os.path.join(self.output_dir, f"{cas}_document.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    return json.load(f)
            results = self.collection.get(
                where={"cas": cas},
                limit=100
            )
            if not results["ids"]:
                logger.warning(f"No se encontró información para el CAS {cas}")
                return None
            chunks_with_ids = list(zip(results["ids"], results["documents"], results["metadatas"]))
            chunks_with_ids.sort(key=lambda x: int(re.search(r'chunk-(\d+)', x[0]).group(1)))
            full_text = "\n\n".join([chunk for _, chunk, _ in chunks_with_ids])
            metadata = chunks_with_ids[0][2]
            return {
                "cas": cas,
                "cid": metadata.get("cid"),
                "name": metadata.get("name", f"Compound {cas}"),
                "formula": metadata.get("formula", ""),
                "weight": metadata.get("molecular_weight", ""),
                "text": full_text,
                "reconstructed_from_chunks": True,
                "total_chunks": len(chunks_with_ids)
            }
        except Exception as e:
            logger.error(f"Error al recuperar información del compuesto {cas}: {str(e)}")
            return None
    
    def get_safety_analysis(self, cas):
        """
        Recupera el análisis de seguridad de un compuesto
        
        Args:
            cas (str): Número CAS del compuesto
            
        Returns:
            dict: Análisis de seguridad o None si no existe
        """
        try:
            safety_path = os.path.join(SAFETY_DIR, f"{cas}_safety_analysis.json")
            if os.path.exists(safety_path):
                with open(safety_path, 'r') as f:
                    return json.load(f)
            logger.warning(f"No se encontró análisis de seguridad para el CAS {cas}")
            return None
        except Exception as e:
            logger.error(f"Error al recuperar análisis de seguridad para CAS {cas}: {str(e)}")
            return None
    
    def get_toxicity_analysis(self, cas):
        """
        Recupera el análisis de toxicidad de un compuesto
        
        Args:
            cas (str): Número CAS del compuesto
            
        Returns:
            dict: Análisis de toxicidad o None si no existe
        """
        try:
            toxicity_path = os.path.join(TOXICITY_DIR, f"{cas}_toxicity_analysis.json")
            if os.path.exists(toxicity_path):
                with open(toxicity_path, 'r') as f:
                    return json.load(f)
            logger.warning(f"No se encontró análisis de toxicidad para el CAS {cas}")
            return None
        except Exception as e:
            logger.error(f"Error al recuperar análisis de toxicidad para CAS {cas}: {str(e)}")
            return None
    
    def export_chemical_database(self, output_csv=None):
        """
        Exporta una base de datos de los productos químicos procesados
        
        Args:
            output_csv (str, optional): Ruta para el archivo CSV de salida
            
        Returns:
            pd.DataFrame: DataFrame con la información de los compuestos
        """
        try:
            results = self.collection.get(
                where={"chunk_id": 0},
                limit=10000
            )
            if not results["ids"]:
                logger.warning("No se encontraron compuestos en la base de datos")
                return pd.DataFrame()
            compounds = []
            for metadata in results["metadatas"]:
                compounds.append({
                    "cas": metadata.get("cas"),
                    "cid": metadata.get("cid"),
                    "name": metadata.get("name"),
                    "formula": metadata.get("formula"),
                    "molecular_weight": metadata.get("molecular_weight", ""),
                    "total_chunks": metadata.get("total_chunks", 1)
                })
            df = pd.DataFrame(compounds)
            if output_csv:
                df.to_csv(output_csv, index=False)
                logger.info(f"Base de datos exportada a {output_csv}")
            return df
        except Exception as e:
            logger.error(f"Error al exportar base de datos: {str(e)}")
            return pd.DataFrame()
    
    def export_chemical_database_with_comptox(self, output_csv=None):
        """
        Exporta una base de datos de los productos químicos procesados incluyendo datos de CompTox
        
        Args:
            output_csv (str, optional): Ruta para el archivo CSV de salida
            
        Returns:
            pd.DataFrame: DataFrame con la información de los compuestos
        """
        try:
            results = self.collection.get(
                where={"chunk_id": 0},
                limit=10000
            )
            if not results["ids"]:
                logger.warning("No se encontraron compuestos en la base de datos")
                return pd.DataFrame()
            compounds = []
            for metadata in results["metadatas"]:
                compound_data = {
                    "cas": metadata.get("cas"),
                    "cid": metadata.get("cid"),
                    "name": metadata.get("name"),
                    "formula": metadata.get("formula"),
                    "molecular_weight": metadata.get("molecular_weight", ""),
                    "total_chunks": metadata.get("total_chunks", 1),
                    "has_comptox_data": metadata.get("has_comptox_data", False)
                }
                for key, value in metadata.items():
                    if key.startswith("comptox_"):
                        compound_data[key] = value
                compounds.append(compound_data)
            df = pd.DataFrame(compounds)
            if output_csv:
                df.to_csv(output_csv, index=False)
                logger.info(f"Base de datos exportada a {output_csv}")
            return df
        except Exception as e:
            logger.error(f"Error al exportar base de datos: {str(e)}")
            return pd.DataFrame()
    
    def recover_embeddings_from_backup(self):
        """
        Recupera embeddings desde archivos de respaldo si la base de datos ChromaDB está vacía
        
        Returns:
            int: Número de embeddings recuperados
        """
        try:
            collection_count = self.collection.count()
            if collection_count > 0:
                logger.info(f"La colección ya tiene {collection_count} documentos, no se necesita recuperación")
                return 0
            logger.info("Iniciando recuperación de embeddings desde respaldos...")
            embeddings_dir = os.path.join(self.output_dir, "embeddings")
            info_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_embedding_info.json')]
            recovered_count = 0
            for info_file in tqdm(info_files, desc="Recuperando embeddings"):
                try:
                    with open(os.path.join(embeddings_dir, info_file), 'r') as f:
                        embedding_info = json.load(f)
                    cas = embedding_info.get('cas')
                    cid = embedding_info.get('cid')
                    chunk_id = embedding_info.get('chunk_id')
                    embedding_file = embedding_info.get('embedding_file')
                    if not all([cas, cid, chunk_id is not None, embedding_file]):
                        logger.warning(f"Información de embedding incompleta en {info_file}")
                        continue
                    embedding = self.load_cached_embedding(embedding_file)
                    if embedding is None:
                        logger.warning(f"No se pudo cargar embedding desde {embedding_file}")
                        continue
                    chunk_path = os.path.join(self.output_dir, "chunks", f"{cas}_chunk_{chunk_id}.txt")
                    if not os.path.exists(chunk_path):
                        logger.warning(f"No se encontró archivo de chunk {chunk_path}")
                        continue
                    with open(chunk_path, 'r') as f:
                        chunk_text = f.read()
                    json_path = os.path.join(self.output_dir, f"{cas}_document.json")
                    name = f"Compound {cid}"
                    formula = ""
                    weight = ""
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            doc_data = json.load(f)
                            name = doc_data.get('name', name)
                            formula = doc_data.get('formula', formula)
                            weight = doc_data.get('weight', weight)
                    chunk_id_str = f"cas-{cas}-chunk-{chunk_id}"
                    self.collection.add(
                        documents=[chunk_text],
                        embeddings=[embedding],
                        metadatas=[{
                            "cas": cas,
                            "cid": str(cid),
                            "chunk_id": chunk_id,
                            "name": name,
                            "formula": formula,
                            "molecular_weight": str(weight),
                            "type": "compound_chunk",
                            "embedding_file": embedding_file,
                            "restored_from_backup": True
                        }],
                        ids=[chunk_id_str]
                    )
                    recovered_count += 1
                    if recovered_count % 10 == 0:
                        try:
                            self.chroma_client.persist()
                            logger.info(f"ChromaDB persistido después de recuperar {recovered_count} embeddings")
                        except Exception as e:
                            logger.warning(f"No se pudo persistir después de recuperar {recovered_count} embeddings: {str(e)}")
                except Exception as e:
                    logger.error(f"Error al recuperar embedding desde {info_file}: {str(e)}")
            try:
                self.chroma_client.persist()
                logger.info(f"Recuperación completada. Se recuperaron {recovered_count} embeddings")
            except Exception as e:
                logger.warning(f"No se pudo persistir al finalizar la recuperación: {str(e)}")
            return recovered_count
        except Exception as e:
            logger.error(f"Error en la recuperación de embeddings: {str(e)}")
            return 0
    
    def comptox_integration_demo(self, cas="71-43-2"):
        """
        Demostración de la integración con CompTox EPA
        
        Args:
            cas (str): Número CAS del compuesto a procesar (por defecto, benceno)
        """
        logger.info(f"=== INICIANDO DEMO DE INTEGRACIÓN COMPTOX EPA PARA CAS {cas} ===")
        comptox_data = self.get_comptox_data_debug(cas)  # Llamada a la versión con debug
        if comptox_data:
            print(f"\n=== DATOS DE COMPTOX EPA PARA CAS {cas} ===\n")
            relevant_fields = ["DTXSID", "PREFERRED_NAME", "CASRN", "Toxicity", 
                              "LogP", "HalfLife", "WaterSolubility"]
            for field in relevant_fields:
                if field in comptox_data and comptox_data[field] and str(comptox_data[field]).lower() != 'nan':
                    print(f"{field}: {comptox_data[field]}")
            print(f"\nTotal de campos disponibles: {len(comptox_data)}")
        else:
            print(f"\nNo se pudieron obtener datos de CompTox EPA para CAS {cas}")
        print(f"\n=== PROCESANDO COMPUESTO CAS {cas} CON DATOS DE COMPTOX ===\n")
        success = self.process_compound(cas)
        if success:
            print(f"Compuesto CAS {cas} procesado correctamente")
            results = self.collection.get(
                where={"cas": cas},
                limit=1
            )
            if results["ids"]:
                metadata = results["metadatas"][0]
                comptox_fields = [k for k in metadata.keys() if k.startswith("comptox_")]
                print(f"\nMetadatos de CompTox en ChromaDB: {comptox_fields}")
            toxicity_analysis = self.get_toxicity_analysis(cas)
            if toxicity_analysis and toxicity_analysis.get("success"):
                print("\n=== ANÁLISIS DE TOXICIDAD CON DATOS DE COMPTOX ===\n")
                print(f"¿Incluye datos de CompTox? {toxicity_analysis.get('comptox_data_included', False)}")
                print("\nExtracto del análisis:")
                print(toxicity_analysis.get("toxicity_analysis", "No disponible")[:500] + "...")
            print("\n=== BÚSQUEDA CON FILTRO DE COMPTOX ===\n")
            search_results = self.search_compounds("toxic effects", n_results=3)
            print(f"Resultados sin filtro: {search_results['total_results']}")
            if comptox_data:
                filter_field = None
                filter_value = None
                for field in ["Toxicity", "PREFERRED_NAME", "LogP"]:
                    if field in comptox_data and comptox_data[field] and str(comptox_data[field]).lower() != 'nan':
                        filter_field = f"comptox_{field}"
                        filter_value = str(comptox_data[field])
                        break
                if filter_field and filter_value:
                    comptox_filter = {filter_field: filter_value}
                    filtered_results = self.search_compounds("toxic effects", n_results=3, comptox_filter=comptox_filter)
                    print(f"Resultados con filtro {filter_field}={filter_value}: {filtered_results['total_results']}")
                    if filtered_results['total_results'] > 0:
                        print("Primer resultado: " + filtered_results['results'][0]['metadata'].get('name', 'Desconocido'))
        print("\n=== EXPORTANDO BASE DE DATOS CON DATOS DE COMPTOX ===\n")
        output_path = os.path.join(OUTPUT_DIR, "chemical_database_with_comptox.csv")
        df = self.export_chemical_database_with_comptox(output_path)
        print(f"La base de datos contiene {len(df)} compuestos")
        comptox_columns = [col for col in df.columns if 'comptox' in col.lower()]
        print(f"Columnas relacionadas con CompTox: {comptox_columns}")
        print("\n=== DEMO DE INTEGRACIÓN COMPTOX EPA COMPLETADA ===\n")

# Funciones de demo y utilidad
def process_single_compound(cas="71-43-2"):
    logger.info(f"=== INICIANDO PROCESAMIENTO DE COMPUESTO {cas} ===")
    analyzer = ChemicalSafetyAnalyzer()
    if analyzer.collection.count() == 0:
        analyzer.recover_embeddings_from_backup()
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
        logger.error(f"Error al procesar el compuesto CAS {cas}")

def process_csv_demo(csv_file, cas_column):
    logger.info(f"=== INICIANDO PROCESAMIENTO DE CSV {csv_file} ===")
    analyzer = ChemicalSafetyAnalyzer()
    if analyzer.collection.count() == 0:
        analyzer.recover_embeddings_from_backup()
    processed_count = analyzer.process_csv_file(csv_file, cas_column)
    logger.info(f"Procesados {processed_count} compuestos del CSV")
    df = analyzer.export_chemical_database(os.path.join(OUTPUT_DIR, "chemical_database.csv"))
    logger.info(f"Base de datos exportada con {len(df)} compuestos")
    print(f"\nSe procesaron {processed_count} compuestos del CSV.")
    print(f"La base de datos contiene {len(df)} compuestos en total.")
    print(f"Datos exportados a {os.path.join(OUTPUT_DIR, 'chemical_database.csv')}")

def interactive_search():
    analyzer = ChemicalSafetyAnalyzer()
    print("\n=== BÚSQUEDA INTERACTIVA DE COMPUESTOS ===\n")
    print("Escribe 'salir' para terminar la búsqueda")
    while True:
        query = input("\nIntroduce tu consulta de búsqueda: ").strip()
        if query.lower() in ['salir', 'exit', 'quit']:
            break
        n_results = input("Número de resultados (por defecto 5): ").strip()
        try:
            n_results = int(n_results) if n_results else 5
        except:
            n_results = 5
        results = analyzer.search_compounds(query, n_results=n_results)
        print(f"\nEncontrados {results['total_results']} resultados para '{query}':")
        for i, result in enumerate(results["results"]):
            print(f"\nResultado {i+1}: {result['metadata'].get('name', 'Desconocido')} (CAS: {result['metadata'].get('cas', 'N/A')})")
            print(f"Similitud: {1 - result['distance']:.2%}")
            print(f"Extracto: {result['document'][:200]}...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analizador de Seguridad Química')
    parser.add_argument('--cas', type=str, default='71-43-2',
                        help='Número CAS del compuesto a procesar (por defecto: 71-43-2, benceno)')
    parser.add_argument('--csv', type=str, help='Ruta al archivo CSV con lista de CAS')
    parser.add_argument('--cas-column', type=str, default='cas',
                        help='Nombre de la columna con números CAS en el CSV (por defecto: cas)')
    parser.add_argument('--search', action='store_true',
                        help='Iniciar búsqueda interactiva')
    parser.add_argument('--comptox-demo', action='store_true',
                        help='Ejecutar demo de integración con CompTox EPA')
    parser.add_argument('--comptox-cas', type=str,
                        help='Número CAS para demo de CompTox (por defecto: valor de --cas)')
    
    args = parser.parse_args()
    
    analyzer = ChemicalSafetyAnalyzer()
    
    # Si la base está vacía, recuperar embeddings desde backup (opcional)
    if analyzer.collection.count() == 0:
        analyzer.recover_embeddings_from_backup()
    
    # Escoger acción
    if args.comptox_demo:
        comptox_cas = args.comptox_cas if args.comptox_cas else args.cas
        analyzer.comptox_integration_demo(comptox_cas)
    elif args.csv:
        process_csv_demo(args.csv, args.cas_column)
    elif args.search:
        interactive_search()
    else:
        process_single_compound(args.cas)
