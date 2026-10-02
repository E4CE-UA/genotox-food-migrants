import pandas as pd

# Cargar el archivo resultante
df = pd.read_excel("genotoxicity_cas_cid.xlsx")

# Contar valores no nulos y únicos
total_filas = len(df)
cas_no_nulos = df['CAS'].notna().sum()
cid_no_nulos = df['PubChem_CID'].notna().sum()
cas_unicos = df['CAS'].dropna().nunique()
cid_unicos = df['PubChem_CID'].dropna().nunique()

print(f"Total de filas: {total_filas}")
print(f"Números CAS no nulos: {cas_no_nulos}")
print(f"IDs PubChem no nulos: {cid_no_nulos}")
print(f"Números CAS únicos: {cas_unicos}")
print(f"IDs PubChem únicos: {cid_unicos}")