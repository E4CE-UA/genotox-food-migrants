import os
import pandas as pd

# Crear un DataFrame simple para pruebas
df = pd.DataFrame({
    'Test': ['This is a test']
})

print("=== SCRIPT DE PRUEBA PARA LOCALIZAR ARCHIVOS ===")
print(f"Directorio de trabajo actual: {os.getcwd()}")

# Imprimir todos los archivos en el directorio actual
print("\nArchivos en el directorio actual:")
for file in os.listdir('.'):
    print(f"  {file}")

# Intentar guardar en diferentes ubicaciones
print("\nIntentando guardar archivos de prueba en diferentes ubicaciones:")

# 1. Directorio actual
try:
    test_file = "test_save_current_dir.xlsx"
    df.to_excel(test_file, index=False)
    abs_path = os.path.abspath(test_file)
    print(f"✓ Archivo guardado con éxito en el directorio actual: {abs_path}")
except Exception as e:
    print(f"✗ Error al guardar en el directorio actual: {str(e)}")

# 2. Directorio de documentos del usuario
try:
    user_docs = os.path.expanduser("~/Documents")
    if os.path.exists(user_docs):
        test_file = os.path.join(user_docs, "test_save_documents.xlsx")
        df.to_excel(test_file, index=False)
        print(f"✓ Archivo guardado con éxito en Documentos: {test_file}")
    else:
        print("✗ No se pudo encontrar el directorio de Documentos del usuario")
except Exception as e:
    print(f"✗ Error al guardar en Documentos: {str(e)}")

# 3. Directorio temporal
try:
    import tempfile
    temp_dir = tempfile.gettempdir()
    test_file = os.path.join(temp_dir, "test_save_temp.xlsx")
    df.to_excel(test_file, index=False)
    print(f"✓ Archivo guardado con éxito en directorio temporal: {test_file}")
except Exception as e:
    print(f"✗ Error al guardar en directorio temporal: {str(e)}")

# 4. Directorio padre
try:
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    test_file = os.path.join(parent_dir, "test_save_parent.xlsx")
    df.to_excel(test_file, index=False)
    print(f"✓ Archivo guardado con éxito en directorio padre: {test_file}")
except Exception as e:
    print(f"✗ Error al guardar en directorio padre: {str(e)}")

print("\n=== BÚSQUEDA DE ARCHIVOS EXISTENTES ===")
# Buscar archivos con patrones similares a los que estamos buscando
import glob

patterns = [
    "*.xlsx",
    "*_with_cas_cid.xlsx", 
    "*_progress_*.xlsx"
]

for pattern in patterns:
    print(f"\nBuscando archivos con patrón: {pattern}")
    files = glob.glob(pattern)
    if files:
        for file in files:
            print(f"  Encontrado: {os.path.abspath(file)}")
    else:
        print(f"  No se encontraron archivos con patrón {pattern} en el directorio actual")

print("\n=== SUGERENCIAS PARA ENCONTRAR TU ARCHIVO ===")
print("1. Intenta buscar en tu sistema por fecha de modificación reciente")
print("2. Revisa si tienes permisos de escritura en el directorio actual")
print("3. Comprueba si hay suficiente espacio en disco")
print("4. Intenta guardar manualmente un Excel en el mismo directorio para ver si funciona")