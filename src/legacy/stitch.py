import sys
import subprocess
import os

# Instalar la versión específica que funciona con el editor
print("Instalando la versión correcta de moviepy...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "moviepy", "-y"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy==1.0.3"])
    print("Moviepy 1.0.3 instalado correctamente")
except Exception as e:
    print(f"Error al instalar moviepy: {e}")

# Ahora importamos moviepy
from openai import OpenAI
import re
import os
import urllib.request
from gtts import gTTS
from moviepy.editor import *  # Esta importación funcionará con la versión 1.0.3
import time  # Para esperar entre operaciones

# Configura tu API key
api_key = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=api_key)

# Función para verificar si un archivo existe y su tamaño
def check_file(file_path):
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"Archivo {file_path} existe. Tamaño: {size} bytes")
        return True, size
    else:
        print(f"Archivo {file_path} no existe")
        return False, 0

# Parte 1: Generación de texto
def generate_text():
    # Set the prompt to generate text for
    text = input("¿Sobre qué tema quieres escribir?: ")
    prompt = text
    print("El BOT de IA está generando un nuevo texto para ti...")
    
    # Usar el endpoint de chat completions que es el recomendado actualmente
    completions = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un asistente útil."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.5,
    )
    
    # Obtener el texto generado
    generated_text = completions.choices[0].message.content
    
    # Guardar el texto en un archivo
    with open("generated_text.txt", "w", encoding="utf-8") as file:
        file.write(generated_text)
    
    # Verificar que el archivo se guardó correctamente
    check_file("generated_text.txt")
    
    print("¡El texto ha sido generado exitosamente!")
    return generated_text

# Parte 2: Creación de video a partir del texto generado
def create_video(text):
    # Dividir el texto por comas y puntos
    paragraphs = re.split(r"[,.]", text)
    
    # Limpiar párrafos vacíos
    paragraphs = [p for p in paragraphs if p.strip()]
    
    # Crear carpetas necesarias y asegurar que existan
    for folder in ["audio", "images", "videos"]:
        os.makedirs(folder, exist_ok=True)
        print(f"Carpeta {folder} creada o ya existe")
    
    # Recorrer cada párrafo y generar una imagen para cada uno
    i = 1
    video_paths = []
    
    for para in paragraphs:
        if para.strip() == "":
            continue
            
        print(f"Procesando párrafo {i}: {para.strip()}")
        
        # Generar imagen con OpenAI
        response = client.images.generate(
            prompt=para.strip(),
            n=1,
            size="1024x1024"
        )
        
        print("Generando nueva imagen de IA desde el párrafo...")
        
        # Descargar la imagen
        image_url = response.data[0].url
        image_path = f"images/image{i}.jpg"
        urllib.request.urlretrieve(image_url, image_path)
        print(f"¡La imagen generada se guardó en {image_path}!")
        
        # Verificar que la imagen se guardó correctamente
        image_exists, image_size = check_file(image_path)
        if not image_exists or image_size == 0:
            print(f"Error: La imagen {image_path} no se guardó correctamente")
            continue
        
        # Crear instancia gTTS y guardar en un archivo
        audio_path = f"audio/voiceover{i}.mp3"
        tts = gTTS(text=para, lang='es', slow=False)
        tts.save(audio_path)
        print(f"¡El párrafo se convirtió en voz y se guardó en {audio_path}!")
        
        # Verificar que el audio se guardó correctamente
        audio_exists, audio_size = check_file(audio_path)
        if not audio_exists or audio_size == 0:
            print(f"Error: El audio {audio_path} no se guardó correctamente")
            continue
        
        # Cargar el archivo de audio usando moviepy
        print("Extrayendo voz y obteniendo duración...")
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        
        # Cargar el archivo de imagen usando moviepy
        print("Extrayendo clip de imagen y estableciendo duración...")
        image_clip = ImageClip(image_path).set_duration(audio_duration)
        
        # Usar moviepy para crear un video final
        print("Creando clip de video...")
        clip = image_clip.set_audio(audio_clip)
        
        # Guardar el video final en un archivo
        video_path = f"videos/video{i}.mp4"
        clip.write_videofile(video_path, fps=24)
        print(f"¡El Video{i} ha sido creado exitosamente!")
        
        # Verificar que el video se guardó correctamente
        video_exists, video_size = check_file(video_path)
        if video_exists and video_size > 0:
            video_paths.append(video_path)
            print(f"Video {video_path} guardado correctamente")
        else:
            print(f"Error: El video {video_path} no se guardó correctamente")
        
        # Cerrar clips para liberar memoria
        audio_clip.close()
        image_clip.close()
        clip.close()
        
        # Esperar un momento antes de continuar
        time.sleep(1)
        
        i += 1
    
    # Combinar todos los clips en un video final
    if video_paths:
        print(f"Intentando concatenar {len(video_paths)} videos...")
        
        clips = []
        for path in video_paths:
            try:
                clip = VideoFileClip(path)
                clips.append(clip)
                print(f"Video {path} cargado correctamente")
            except Exception as e:
                print(f"Error al cargar {path}: {e}")
        
        print("Concatenando todos los clips para crear un video final...")
        if clips:
            try:
                # Especificar ruta absoluta para el video final
                final_video_path = os.path.abspath("final_video.mp4")
                print(f"Guardando video final en: {final_video_path}")
                
                final_video = concatenate_videoclips(clips, method="compose")
                final_video.write_videofile(final_video_path, fps=24)
                
                # Verificar que el video final se guardó correctamente
                final_exists, final_size = check_file(final_video_path)
                if final_exists and final_size > 0:
                    print(f"¡El video final ha sido creado exitosamente en {final_video_path}!")
                    print(f"Tamaño del video: {final_size} bytes")
                    # Intentar abrir el video automáticamente
                    try:
                        if sys.platform == "darwin":  # macOS
                            subprocess.run(["open", final_video_path])
                        elif sys.platform == "win32":  # Windows
                            os.startfile(final_video_path)
                        else:  # Linux
                            subprocess.run(["xdg-open", final_video_path])
                    except Exception as e:
                        print(f"No se pudo abrir el video automáticamente: {e}")
                else:
                    print(f"Error: El video final no se guardó correctamente")
                    
                # Intentar crear el video final usando ffmpeg directamente como backup
                try:
                    print("Intentando crear video final con ffmpeg como método alternativo...")
                    # Crear lista de archivos
                    with open("videos/file_list.txt", "w") as f:
                        for video_path in video_paths:
                            # Solo el nombre del archivo, no la ruta completa
                            video_file = os.path.basename(video_path)
                            f.write(f"file '{video_file}'\n")
                    
                    # Usar ffmpeg para concatenar
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
                        "-i", "file_list.txt", "-c", "copy", 
                        "../final_video_ffmpeg.mp4"
                    ]
                    subprocess.run(ffmpeg_cmd, check=True, cwd="videos")
                    print("¡Video alternativo creado con ffmpeg en final_video_ffmpeg.mp4!")
                    check_file("final_video_ffmpeg.mp4")
                except Exception as e:
                    print(f"Error al crear video con ffmpeg: {e}")
                
                # Cerrar clips
                for clip in clips:
                    clip.close()
                final_video.close()
                
            except Exception as e:
                print(f"Error al crear video final: {e}")
        else:
            print("No hay clips para concatenar")
    else:
        print("No se encontraron videos para concatenar")
    
    # Mostrar todos los archivos generados
    print("\nListado de todos los archivos generados:")
    for folder in [".", "videos", "images", "audio"]:
        if os.path.exists(folder):
            print(f"\nArchivos en {folder}:")
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    print(f"- {file} ({os.path.getsize(file_path)} bytes)")

# Ejecutar el programa
if __name__ == "__main__":
    try:
        # Generar texto
        generated_text = generate_text()
        
        # Crear video a partir del texto generado
        create_video(generated_text)
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")