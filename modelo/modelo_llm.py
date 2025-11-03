import sys
import time
import threading

from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- CONSTANTES DE CONFIGURACIÓN ---
CHROMA_PATH = "chroma_db_web"
MODELO_OLLAMA = "phi3:mini"
MODELO_EMBEDDING = "nomic-embed-text"

# --- FUNCIÓN DE CONFIGURACIÓN DE LA CADENA RAG ---
def get_rag_chain():
    """
    Configura y devuelve la cadena de RAG (Retrieval-Augmented Generation) completa.
    Si ocurre un error durante la configuración (ej. no se encuentra Chroma),
    la excepción será lanzada para que la función que llama la maneje.
    """
    embeddings = OllamaEmbeddings(model=MODELO_EMBEDDING)
    
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    template ="""
    Actúa como un asistente virtual experto y muy servicial de la Universidad Veracruzana. 
    Tu misión es proporcionar respuestas extremadamente detalladas y completas, utilizando únicamente la información encontrada en el CONTEXTO proporcionado.

    Sigue estas reglas estrictamente:
    1.  **Sé Exhaustivo:** Extrae y sintetiza TODA la información relevante del contexto que responda a la pregunta del usuario. No omitas detalles, requisitos, fechas o pasos mencionados.
    2.  **Organiza la Información:** Estructura tu respuesta de una manera clara y fácil de entender. Si la pregunta es sobre un proceso, descríbelo en una lista ordenada (paso a paso). Si se listan requisitos, usa viñetas.
    3.  **Elabora la Respuesta:** No te limites a extraer texto. Explica los conceptos con tus propias words (basadas en el contexto) para que la respuesta sea coherente y completa. El objetivo es que el usuario entienda el tema a fondo.
    4.  **Restricción Absoluta:** Si la información necesaria para responder la pregunta no se encuentra en el CONTEXTO, DEBES responder única y exclusivamente con la frase: "No tengo información suficiente sobre eso en mis documentos." No intentes adivinar ni añadir información externa.

    ---
    CONTEXTO:
    {context}
    ---
    PREGUNTA DEL USUARIO:
    {question}
    ---

    RESPUESTA DETALLADA Y COMPLETA:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model=MODELO_OLLAMA)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- FUNCIÓN PARA LA ANIMACIÓN DE CARGA ---
def mostrar_barra_de_carga(evento_parada):
    """
    Muestra una barra de progreso animada en la consola hasta que
    el 'evento_parada' se active.
    """
    bar_length = 30
    i = 0
    while not evento_parada.is_set():
        progreso = i % 100
        longitud_llena = int(bar_length * progreso // 100)
        barra = '█' * longitud_llena + '-' * (bar_length - longitud_llena)
        sys.stdout.write(f'\rChatbot: Procesando... |{barra}| {progreso}%')
        sys.stdout.flush()
        i += 2
        time.sleep(0.1)

    barra_final = '█' * bar_length
    sys.stdout.write(f'\rChatbot: Procesando... |{barra_final}| 100%\n')
    sys.stdout.flush()

# --- NO HAY FUNCIÓN MAIN ---
# Este archivo ahora es solo un módulo para ser importado.