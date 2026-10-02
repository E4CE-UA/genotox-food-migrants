"""
resolver.py — identificadores -> estructura, a partir de la logica de cas.py y cid_cas_blanca.py

Orden de resolucion por fila (el primero que funcione gana):
  1. PubChem_CID que ya viene en el Excel de EFSA
  2. CAS  -> CID   (pug/compound/xref/RN/{cas}/cids)         <- tu get_cid_from_cas
  3. Nombre -> CID (pug/compound/name/{nombre}/cids)         <- tu cid_cas_blanca
Luego, en lotes de 100:  CID -> SMILES, InChIKey, formula, peso molecular

Mejoras sobre los scripts originales:
  - cache en disco (cache/*.json): si se corta, al relanzar sigue donde iba
  - reintentos con espera exponencial ante 429/5xx (tu make_request_with_retry)
  - propiedades por lotes de 100 CIDs (antes, 1 peticion por compuesto)
  - decodifica los escapes de Excel en los nombres (_x0028_ -> "(")
  - NO estandariza ni quita fragmentos: guarda el SMILES completo
    (eso se decide despues, sabiendo si es sal, mezcla, polimero...)

Uso:
  python src/resolver.py --test      # prueba rapida con benceno
  python src/resolver.py             # dataset EFSA completo
"""

import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "data" / "genotoxicity_cas_cid.xlsx"
SALIDA = RAIZ / "data" / "efsa_estructuras.csv"
CACHE_DIR = RAIZ / "cache"

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PAUSA = 0.25          # PubChem admite ~5 peticiones por segundo
LOTE = 100
REINTENTOS = 4
# PubChem renombro las propiedades SMILES en 2025; probamos nuevas y antiguas
JUEGOS_PROPIEDADES = [
    "SMILES,InChIKey,MolecularFormula,MolecularWeight",
    "IsomericSMILES,InChIKey,MolecularFormula,MolecularWeight",
    "CanonicalSMILES,InChIKey,MolecularFormula,MolecularWeight",
]
PATRON_CAS = re.compile(r"^\d{2,7}-\d{2}-\d$")


# ---------------------------------------------------------------- utilidades
def decodificar_excel(texto):
    """_x0028_ -> '(' , _x0020_ -> ' ' , etc. (escapes XML de Excel)."""
    if not isinstance(texto, str):
        return texto
    return re.sub(r"_x([0-9A-Fa-f]{4})_", lambda m: chr(int(m.group(1), 16)), texto).strip()


def cas_valido(cas):
    """Comprueba formato y digito de control del numero CAS."""
    if not isinstance(cas, str) or not PATRON_CAS.match(cas.strip()):
        return False
    digitos = cas.replace("-", "")
    cuerpo, control = digitos[:-1], int(digitos[-1])
    suma = sum(int(d) * (i + 1) for i, d in enumerate(reversed(cuerpo)))
    return suma % 10 == control


class Cache:
    """Diccionario persistente en un JSON. Guarda cada N escrituras."""

    def __init__(self, nombre):
        CACHE_DIR.mkdir(exist_ok=True)
        self.ruta = CACHE_DIR / f"{nombre}.json"
        self.datos = json.loads(self.ruta.read_text()) if self.ruta.exists() else {}
        self._pendientes = 0

    def __contains__(self, k):
        return str(k) in self.datos

    def get(self, k):
        return self.datos.get(str(k))

    def set(self, k, v):
        self.datos[str(k)] = v
        self._pendientes += 1
        if self._pendientes >= 25:
            self.guardar()

    def guardar(self):
        self.ruta.write_text(json.dumps(self.datos))
        self._pendientes = 0


def peticion(url):
    """GET con reintentos y espera exponencial. Devuelve JSON o None."""
    for intento in range(REINTENTOS):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                time.sleep(PAUSA)
                return r.json()
            if r.status_code == 404:          # PubChem: "no encontrado"
                time.sleep(PAUSA)
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep((2 ** intento) * 0.5)
                continue
            return {"_error": r.status_code}  # 400 u otro: no reintentar
        except requests.RequestException:
            time.sleep((2 ** intento) * 0.5)
    return None


def primer_cid(json_respuesta):
    try:
        return int(json_respuesta["IdentifierList"]["CID"][0])
    except (TypeError, KeyError, IndexError, ValueError):
        return None


# ------------------------------------------------------- identificador -> CID
_cas2cid = Cache("cas2cid")
_nombre2cid = Cache("nombre2cid")
_propiedades = Cache("cid_propiedades")


def cid_desde_cas(cas):
    cas = cas.strip()
    if cas in _cas2cid:
        return _cas2cid.get(cas)
    cid = primer_cid(peticion(f"{PUG}/compound/xref/RN/{cas}/cids/JSON"))
    _cas2cid.set(cas, cid)
    return cid


def cid_desde_nombre(nombre):
    nombre = decodificar_excel(nombre)
    if not nombre:
        return None
    if nombre in _nombre2cid:
        return _nombre2cid.get(nombre)
    url = f"{PUG}/compound/name/{urllib.parse.quote(nombre, safe='')}/cids/JSON"
    cid = primer_cid(peticion(url))
    _nombre2cid.set(nombre, cid)
    return cid


# ------------------------------------------------------ CID -> propiedades
def propiedades_por_lotes(cids):
    """Rellena la cache con SMILES/InChIKey/formula/PM de todos los CIDs."""
    pendientes = [c for c in dict.fromkeys(cids) if c is not None and c not in _propiedades]
    print(f"  propiedades pendientes: {len(pendientes)} CIDs")
    juego = 0
    for i in range(0, len(pendientes), LOTE):
        lote = pendientes[i:i + LOTE]
        url = f"{PUG}/compound/cid/{','.join(map(str, lote))}/property/{JUEGOS_PROPIEDADES[juego]}/JSON"
        datos = peticion(url)
        # si PubChem rechaza el nombre de la propiedad (400), probamos el siguiente juego
        while isinstance(datos, dict) and "_error" in datos and juego < len(JUEGOS_PROPIEDADES) - 1:
            juego += 1
            print(f"  PubChem rechazo el juego de propiedades; probando: {JUEGOS_PROPIEDADES[juego]}")
            url = f"{PUG}/compound/cid/{','.join(map(str, lote))}/property/{JUEGOS_PROPIEDADES[juego]}/JSON"
            datos = peticion(url)
        filas = (datos or {}).get("PropertyTable", {}).get("Properties", []) if isinstance(datos, dict) else []
        recibidos = set()
        for p in filas:
            smiles = p.get("SMILES") or p.get("IsomericSMILES") or p.get("CanonicalSMILES")
            _propiedades.set(p["CID"], {
                "smiles": smiles,
                "inchikey": p.get("InChIKey"),
                "formula": p.get("MolecularFormula"),
                "peso_molecular": p.get("MolecularWeight"),
            })
            recibidos.add(p["CID"])
        for c in lote:                       # CIDs sin respuesta: se marcan para no repetir
            if c not in recibidos:
                _propiedades.set(c, None)
        print(f"  {min(i + LOTE, len(pendientes))}/{len(pendientes)}", end="\r", flush=True)
    _propiedades.guardar()
    print()


def resolver(cas=None, nombre=None, cid=None):
    """Resuelve un compuesto suelto. Pensado para el widget del notebook."""
    fuente = None
    if cid:
        fuente = "cid"
    elif cas and cas_valido(str(cas)):
        cid, fuente = cid_desde_cas(str(cas)), "cas"
    if not cid and nombre:
        cid, fuente = cid_desde_nombre(nombre), "nombre"
    if not cid:
        return {"cid": None, "fuente": None}
    cid = int(cid)
    if cid not in _propiedades:
        propiedades_por_lotes([cid])
    return {"cid": cid, "fuente": fuente, **(_propiedades.get(cid) or {})}


# ------------------------------------------------------------- dataset EFSA
def procesar_efsa():
    df = pd.read_excel(ENTRADA)
    for col in ["Substance", "Author", "Genotoxicity", "CleanSubstanceName"]:
        df[col] = df[col].map(decodificar_excel)
    print(f"Filas: {len(df)}")

    df["cid"] = pd.to_numeric(df["PubChem_CID"], errors="coerce").astype("Int64")
    df["fuente_cid"] = df["cid"].notna().map({True: "excel", False: None})

    # 2. por CAS, una consulta por CAS unico
    sin = df["cid"].isna() & df["CAS"].map(lambda x: cas_valido(str(x)) if pd.notna(x) else False)
    cas_unicos = df.loc[sin, "CAS"].str.strip().unique()
    print(f"CAS -> CID: {len(cas_unicos)} CAS unicos")
    try:
        for n, cas in enumerate(cas_unicos, 1):
            cid_desde_cas(cas)
            print(f"  {n}/{len(cas_unicos)}", end="\r", flush=True)
    finally:
        _cas2cid.guardar()
    print()
    m = df["cid"].isna() & df["CAS"].notna()
    df.loc[m, "cid"] = df.loc[m, "CAS"].map(lambda c: _cas2cid.get(str(c).strip())).astype("Int64")
    df.loc[m & df["cid"].notna(), "fuente_cid"] = "cas"

    # 3. por nombre, lo que quede
    nombres = df.loc[df["cid"].isna(), "CleanSubstanceName"].dropna().unique()
    print(f"Nombre -> CID: {len(nombres)} nombres unicos")
    try:
        for n, nom in enumerate(nombres, 1):
            cid_desde_nombre(nom)
            print(f"  {n}/{len(nombres)}", end="\r", flush=True)
    finally:
        _nombre2cid.guardar()
    print()
    m = df["cid"].isna()
    df.loc[m, "cid"] = df.loc[m, "CleanSubstanceName"].map(lambda x: _nombre2cid.get(x)).astype("Int64")
    df.loc[m & df["cid"].notna(), "fuente_cid"] = "nombre"

    # 4. propiedades por lotes
    print("CID -> estructura")
    propiedades_por_lotes([int(c) for c in df["cid"].dropna().unique()])
    props = df["cid"].map(lambda c: _propiedades.get(int(c)) if pd.notna(c) else None)
    for campo in ["smiles", "inchikey", "formula", "peso_molecular"]:
        df[campo] = props.map(lambda p: p.get(campo) if isinstance(p, dict) else None)
    # numero de fragmentos (1 = molecula unica; >1 = sal, mezcla o complejo)
    df["n_fragmentos"] = df["smiles"].map(lambda s: s.count(".") + 1 if isinstance(s, str) else None)

    df.to_csv(SALIDA, index=False)
    resumen(df)


def resumen(df):
    print(f"\n=== {SALIDA.name} ===")
    print(f"filas con estructura  : {df['smiles'].notna().sum()} de {len(df)} "
          f"({df['smiles'].notna().mean():.1%})")
    print(f"estructuras unicas    : {df['inchikey'].nunique()}")
    print("\norigen del CID:")
    print(df["fuente_cid"].value_counts(dropna=False).to_string())
    print("\nfragmentos por estructura:")
    print(df["n_fragmentos"].value_counts().sort_index().head(6).to_string())
    print("\nveredictos con estructura:")
    print(df.loc[df["smiles"].notna(), "Genotoxicity"].value_counts().to_string())


if __name__ == "__main__":
    if "--test" in sys.argv:
        print("CAS 71-43-2 ->", resolver(cas="71-43-2"))
        print("nombre 'benzene' ->", resolver(nombre="benzene"))
        print("decodificar ->", decodificar_excel("_x0028__x002B__x0029_-Lupanine"))
        print("CAS valido 71-43-2:", cas_valido("71-43-2"), "| 71-43-3:", cas_valido("71-43-3"))
    else:
        procesar_efsa()
