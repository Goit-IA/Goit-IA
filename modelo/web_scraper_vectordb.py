# --- web_scraper_vectordb_mejorado.py ---

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import traceback # Importamos la librería para obtener detalles del error

# --- CONSTANTES DE CONFIGURACIÓN ---

# ¡IMPORTANTE! Aquí debes poner la lista de URLs que quieres procesar.
URLS_A_ESCANEAR = [
    "https://www.uv.mx/estudiantes/tramites-escolares/",
    "https://www.uv.mx/estudiantes/tramites-escolares/tramites-escolares-total/",
    "https://www.uv.mx/estudiantes/tramites-escolares/inscripcion-academico-administrativa/",
    "https://www.uv.mx/estudiantes/tramites-escolares/credencial-estudiante-fisica/",
    "https://www.uv.mx/estudiantes/tramites-escolares/credencial-estudiante-digital/",
    "https://www.uv.mx/estudiantes/tramites-escolares/seguro-facultativo/",
    "https://www.uv.mx/estudiantes/tramites-escolares/examen-de-salud-integral-esi/",
    "https://www.uv.mx/estudiantes/tramites-escolares/cambio-programa-educativo/",
    "https://www.uv.mx/estudiantes/tramites-escolares/reinscripcion-inscripcion-en-linea-il/",
    "https://www.uv.mx/estudiantes/tramites-escolares/declaracion-de-equivalencia-o-revalidacion-de-estudios/",
    "https://www.uv.mx/estudiantes/tramites-escolares/baja-temporal-por-periodo-escolar/",
    "https://www.uv.mx/estudiantes/tramites-escolares/baja-temporal-por-experiencia-educativa/",
    "https://www.uv.mx/estudiantes/tramites-escolares/baja-temporal-extemporanea/",
    "https://www.uv.mx/estudiantes/tramites-escolares/baja-definitiva/",
    "https://www.uv.mx/estudiantes/tramites-escolares/cambio-tutor-academico/",
    "https://www.uv.mx/estudiantes/tramites-escolares/traslado-escolar/",
    "https://www.uv.mx/estudiantes/tramites-escolares/dictamen-para-la-acreditacion-del-idioma-ingles-o-acreditacion-de-la-lengua/",
    "https://www.uv.mx/estudiantes/tramites-escolares/transferencia-de-calificacion-de-ee-a-otro-programa-educativo/",
    "https://www.uv.mx/estudiantes/tramites-escolares/equivalencia-y-o-transferencia-de-calificacion-de-lengua-i-ii-o-ingles-i-ii-al-mismo-programa-educativo-que-cursa-el-alumno/",
    "https://www.uv.mx/estudiantes/tramites-escolares/reconocimiento-de-creditos-por-ee-acreditadas-en-programas-educativos-cursados-previamente/",
    "https://www.uv.mx/estudiantes/tramites-escolares/transferencia-de-creditos-para-el-afel-a-traves-de-las-ee-de-centros-de-idiomas/",
    "https://www.uv.mx/estudiantes/tramites-escolares/movilidad-estudiantil-institucional/",
    "https://www.uv.mx/estudiantes/tramites-escolares/movilidad-estudiantil-nacional/",
    "https://www.uv.mx/estudiantes/tramites-escolares/movilidad-estudiantil-internacional/",
    "https://www.uv.mx/estudiantes/tramites-escolares/cumplimiento-de-servicio-social/",
    "https://www.uv.mx/estudiantes/tramites-escolares/acreditacion-de-la-experiencia-recepcional/",
    "https://www.uv.mx/estudiantes/tramites-escolares/certificado-de-estudios-completo-o-incompleto/",
    "https://www.uv.mx/estudiantes/tramites-escolares/legalizacion-de-certificados-de-estudio/",
    "https://www.uv.mx/estudiantes/tramites-escolares/expedicion-de-titulo-diploma-y-grado-academico/",
    "https://www.uv.mx/estudiantes/tramites-escolares/cedula-profesional/",
    "http://subsegob.veracruz.gob.mx/documentos.php",
    "https://www.uv.mx/estudiantes/tramites-escolares/registro-de-inicio-y-liberacion-del-servicio-social/",
    "https://www.uv.mx/estudiantes/tramites-escolares/autorizacion-de-examen-profesional-o-exencion/",
    "https://www.uv.mx/estudiantes/tramites-escolares/expedicion-de-carta-de-pasante/",
    "https://www.uv.mx/estudiantes/tramites-escolares/autorizacion-de-examen-de-grado/",
    "https://www.uv.mx/dgrf/files/2025/02/Tabulador-de-Cuotas-por-Serv.-Acad.-y-Admvos.-UV-febrero-2025.pdf",
    "https://www.uv.mx/dgae/circulares/",
    "https://www.uv.mx/secretariaacademica/cuotas-del-comite-pro-mejoras/",
    "https://www.uv.mx/legislacion/files/2023/01/RComite%CC%81sProMejoras2023.pdf",
    "https://www.uv.mx/transparencia/ot875/comite-pro-mejoras/",
    "https://www.uv.mx/orizaba/negocios/informes-de-situacion-financiera/",
    "https://www.uv.mx/secretariaacademica/files/2024/09/CIRCULAR-005-2023.pdf",
    "https://www.uv.mx/secretariaacademica/files/2024/11/lineamientos-cuotas-2025.pdf"
]

CHROMA_PATH = "chroma_db_web" 
MODELO_EMBEDDING = "nomic-embed-text" 

def raspar_y_limpiar_urls(urls):
    """
    Recopila el contenido de una lista de URLs, lo limpia y lo convierte
    en una lista de objetos Document de LangChain.
    """
    print("Iniciando el proceso de web scraping...")
    documentos_procesados = []

    for url in urls:
        try:
            print(f"Procesando: {url}")
            
            # --- MEJORA 1: Identificar y saltar PDFs ---
            if url.lower().endswith('.pdf'):
                print(f"-> Omitiendo URL porque es un archivo PDF: {url}")
                continue # Pasa a la siguiente URL

            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()

            texto_crudo = soup.get_text()
            
            lineas = (line.strip() for line in texto_crudo.splitlines())
            fragmentos = (frase.strip() for line in lineas for frase in line.split("  "))
            texto_limpio = '\n'.join(f for f in fragmentos if f)

            if texto_limpio:
                nuevo_documento = Document(page_content=texto_limpio, metadata={"source": url})
                documentos_procesados.append(nuevo_documento)
        
        # --- MEJORA 2: Capturar CUALQUIER error para evitar que el programa se cierre ---
        except Exception as e:
            print(f"ERROR: No se pudo procesar la URL {url}.")
            print(f"   Motivo: {e}")
            # traceback.print_exc() # Descomenta esta línea para ver un error mucho más detallado

    print(f"\nSe procesaron exitosamente {len(documentos_procesados)} páginas web.")
    return documentos_procesados

def dividir_documentos(documentos):
    if not documentos:
        return None
    print("Dividiendo documentos en fragmentos (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len,
        add_start_index=True
    )
    fragmentos = text_splitter.split_documents(documentos)
    print(f"Los documentos se dividieron en {len(fragmentos)} fragmentos.")
    return fragmentos

def crear_y_guardar_vectordb(fragmentos):
    if not fragmentos:
        print("No hay fragmentos para procesar.")
        return

    try:
        print("Creando la base de datos vectorial con ChromaDB...")
        embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
        vectorstore = Chroma.from_documents(
            documents=fragmentos,
            embedding=embeddings,
            persist_directory=CHROMA_PATH
        )
        print(f"¡Base de datos guardada exitosamente en la carpeta '{CHROMA_PATH}'!")
    
    # --- MEJORA 3: Capturar error si Ollama no está disponible ---
    except Exception as e:
        print("\n--- ERROR CRÍTICO ---")
        print("No se pudo crear la base de datos vectorial.")
        print("Motivo:", e)
        print("\nPosibles soluciones:")
        print("1. Asegúrate de que Ollama esté instalado y en ejecución.")
        print("2. Abre otra terminal y ejecuta el comando 'ollama serve'.")
        print("3. Verifica que el modelo 'nomic-embed-text' esté descargado ('ollama pull nomic-embed-text').")
        print("---------------------\n")


# --- BLOQUE DE EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    documentos_web = raspar_y_limpiar_urls(URLS_A_ESCANEAR)
    
    if documentos_web:
        fragmentos_de_texto = dividir_documentos(documentos_web)
        
        if fragmentos_de_texto:
            crear_y_guardar_vectordb(fragmentos_de_texto)
            
    print("\nProceso completado. Tu base de datos vectorial está lista para ser usada.")