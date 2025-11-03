# --- generador_faq.py ---

import csv
import random
import json
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM as Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm # Importamos tqdm para una barra de progreso
import sys
import traceback

# --- CONSTANTES DE CONFIGURACIÓN ---

# Configuración de la Base de Datos (debe coincidir con tu script anterior)
CHROMA_PATH = "chroma_db_web" 
MODELO_EMBEDDING = "nomic-embed-text" 

# Configuración del LLM de Generación
# ¡IMPORTANTE! Asegúrate de tener este modelo descargado (ej. 'ollama pull llama3')
MODELO_LLM = "llama3" 

# Configuración de Salida
ARCHIVO_CSV_SALIDA = "faq.csv"
NUM_PREGUNTAS_A_GENERAR = 1010 # Más de 1000

# Temas semilla para iniciar la búsqueda de contexto
# Estos temas guían al retriever para encontrar fragmentos de información relevante
TEMAS_SEMENTE = [
    "inscripción", "reinscripción", "servicio social", "trámites de titulación",
    "cuotas y pagos", "aranceles", "credencial de estudiante", "baja temporal",
    "baja definitiva", "movilidad estudiantil", "examen de salud integral",
    "seguro facultativo", "cambio de programa educativo", "equivalencia de estudios",
    "traslado escolar", "acreditación de idioma", "experiencia recepcional",
    "certificado de estudios", "cédula profesional", "carta de pasante",
    "comité pro mejoras", "movilidad nacional", "movilidad internacional"
]

# --- PLANTILLA DE PROMPT ---

# Este prompt es crucial. Le da al LLM su personalidad y sus instrucciones.
PROMPT_TEMPLATE = """
Eres un asistente experto en la creación de conjuntos de datos para chatbots universitarios.
Tu objetivo es generar un par de pregunta y respuesta basado *estrictamente* en el contexto proporcionado.

PERSONA DEL ESTUDIANTE (para la pregunta):
El estudiante que hace la pregunta pertenece a la "Facultad de Negocios y Tecnologías" y está interesado en datos académicos, trámites y costos.

CONTEXTO (Información de la base de datos):
---
{contexto}
---

INSTRUCCIONES:
1.  Basándote *únicamente* en el contexto anterior, genera una (1) pregunta realista que haría el estudiante descrito.
2.  Genera una (1) respuesta clara y concisa a esa pregunta, usando *solo* la información del contexto.
3.  Si el contexto es muy pobre, genera una pregunta sobre el tema general y una respuesta breve indicando que los detalles se deben consultar en ventanilla.
4.  Tu respuesta DEBE estar en formato JSON, con las claves "pregunta" y "respuesta".
5.  No añadas nada antes ni después del JSON.

EJEMPLO DE SALIDA:
{
  "pregunta": "¿Cuál es el costo de la credencial de estudiante física?",
  "respuesta": "El costo de reposición de la credencial de estudiante física es de $X.XX MXN."
}

RESPUESTA JSON (solo el JSON):
"""

def cargar_base_de_datos():
    """
    Carga la base de datos vectorial de ChromaDB existente.
    """
    print(f"Cargando base de datos desde: {CHROMA_PATH}")
    try:
        embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
        vector_store = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embeddings
        )
        # Creamos un "retriever" que busca los 'k' fragmentos más similares
        # k=3 significa que buscará los 3 fragmentos más relevantes
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        print("¡Base de datos cargada exitosamente!")
        return retriever
    except Exception as e:
        print("\n--- ERROR CRÍTICO AL CARGAR LA BASE DE DATOS ---")
        print(f"No se pudo cargar la base de datos desde '{CHROMA_PATH}'.")
        print("Motivo:", e)
        print("\nPosibles soluciones:")
        print("1. Asegúrate de haber ejecutado primero el script 'web_scraper_vectordb.py'.")
        print("2. Verifica que la carpeta 'chroma_db_web' exista en el mismo directorio.")
        print("--------------------------------------------------\n")
        return None

def crear_cadena_generacion(llm):
    """
    Crea la cadena de LangChain (LCEL) que conecta el prompt, el LLM y el parser.
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Esta cadena toma el 'contexto', lo pasa al prompt,
    # luego al LLM, y finalmente limpia la salida.
    cadena = (
        prompt
        | llm
        | StrOutputParser()
    )
    return cadena

def formatear_contexto(documentos):
    """
    Combina el contenido de varios documentos en un solo bloque de texto.
    """
    return "\n\n---\n\n".join(doc.page_content for doc in documentos)

def generar_datos_faq(retriever, chain, num_preguntas):
    """
    Bucle principal para generar la cantidad deseada de preguntas y respuestas.
    """
    lista_faq = []
    print(f"Iniciando la generación de {num_preguntas} pares de Q&A...")
    errores_json = 0

    # Usamos tqdm para mostrar una barra de progreso
    with tqdm(total=num_preguntas, desc="Generando Q&A") as pbar:
        while len(lista_faq) < num_preguntas:
            try:
                # 1. Elegir un tema semilla al azar
                topico_aleatorio = random.choice(TEMAS_SEMENTE)
                
                # 2. Obtener contexto relevante de la base de datos
                documentos_contexto = retriever.invoke(topico_aleatorio)
                
                # Si no hay contexto, saltamos esta iteración
                if not documentos_contexto:
                    continue
                    
                contexto_formateado = formatear_contexto(documentos_contexto)
                
                # 3. Invocar la cadena con el contexto
                respuesta_llm = chain.invoke({"contexto": contexto_formateado})
                
                # 4. Intentar parsear la respuesta JSON
                # A veces el LLM puede agregar texto extra, intentamos limpiarlo
                json_str = respuesta_llm.strip().replace("```json", "").replace("```", "")
                
                datos_qa = json.loads(json_str)
                
                # 5. Validar y añadir a la lista
                if "pregunta" in datos_qa and "respuesta" in datos_qa:
                    if datos_qa["pregunta"] and datos_qa["respuesta"]: # Nos aseguramos que no estén vacíos
                        lista_faq.append(datos_qa)
                        pbar.update(1) # Actualizamos la barra de progreso solo si fue exitoso
                
            except json.JSONDecodeError:
                errores_json += 1
                # Opcional: imprimir el error para depuración
                # tqdm.write(f"ADVERTENCIA: El LLM devolvió un JSON inválido. Omitiendo.")
                # tqdm.write(f"   Respuesta recibida: {respuesta_llm[:100]}...")
            except Exception as e:
                tqdm.write(f"ERROR inesperado durante la generación: {e}")
                traceback.print_exc() # Muestra el error detallado

    print(f"\n¡Generación completada!")
    print(f"Se crearon {len(lista_faq)} pares de Q&A.")
    print(f"Se encontraron {errores_json} errores de formato JSON (omitidos).")
    return lista_faq

def guardar_en_csv(lista_faq, archivo_salida):
    """
    Guarda la lista de diccionarios (pregunta/respuesta) en un archivo CSV.
    """
    if not lista_faq:
        print("No se generaron datos, no se guardará ningún archivo CSV.")
        return

    print(f"Guardando datos en '{archivo_salida}'...")
    try:
        with open(archivo_salida, 'w', newline='', encoding='utf-8') as f:
            # Definimos las columnas que queremos en el CSV
            fieldnames = ['pregunta', 'respuesta']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader() # Escribe la fila de encabezado (pregunta,respuesta)
            
            for qa_par in lista_faq:
                writer.writerow(qa_par)
                
        print(f"¡Archivo '{archivo_salida}' guardado exitosamente!")
    except IOError as e:
        print(f"ERROR: No se pudo escribir en el archivo '{archivo_salida}'.")
        print("Motivo:", e)

# --- BLOQUE DE EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    
    # 1. Cargar la base de datos vectorial
    retriever = cargar_base_de_datos()
    
    if retriever:
        try:
            # 2. Inicializar el LLM
            print(f"Inicializando LLM: {MODELO_LLM} (esto puede tardar un momento)...")
            # Le pedimos al LLM que fuerce la salida en formato JSON
            llm = Ollama(model=MODELO_LLM, format="json") 
            
            # Probamos la conexión con Ollama
            print("Probando conexión con Ollama...")
            llm.invoke("Hola") 
            print("¡Conexión con Ollama y LLM exitosa!")

            # 3. Crear la cadena de generación
            cadena_qa = crear_cadena_generacion(llm)
            
            # 4. Iniciar el proceso de generación
            datos_faq_generados = generar_datos_faq(retriever, cadena_qa, NUM_PREGUNTAS_A_GENERAR)
            
            # 5. Guardar los resultados en CSV
            guardar_en_csv(datos_faq_generados, ARCHIVO_CSV_SALIDA)

        except Exception as e:
            print("\n--- ERROR CRÍTICO ---")
            print("No se pudo conectar con el modelo de Ollama.")
            print("Motivo:", e)
            print("\nPosibles soluciones:")
            print("1. Asegúrate de que Ollama esté instalado y en ejecución.")
            print("2. Abre otra terminal y ejecuta el comando 'ollama serve'.")
            print(f"3. Verifica que el modelo '{MODELO_LLM}' esté descargado ('ollama pull {MODELO_LLM}').")
            print("---------------------\n")
            
    print("\nProceso finalizado.")