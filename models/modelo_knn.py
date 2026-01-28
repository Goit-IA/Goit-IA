import pandas as pd
import re
import os
import sys
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# --- IMPORTS DE BASE DE DATOS (NUEVO) ---
# Aseguramos que la raíz del proyecto esté en el path para importar 'database'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

from database import engine

# --- INICIALIZACIÓN GLOBAL ---

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words_global = set(stopwords.words('spanish'))

# Variables globales del modelo
knn_model = None
vectorizer = None
respuestas_knn = []

def limpiar_texto(texto):
    """Limpia y preprocesa una cadena de texto."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palabras = texto.split()
    palabras_filtradas = [p for p in palabras if p not in stop_words_global]
    return ' '.join(palabras_filtradas)

def inicializar_knn():
    """
    Carga los datos desde PostgreSQL y entrena (o re-entrena) el modelo KNN.
    Esta función se llamará al inicio y cada vez que el chatbot aprenda algo nuevo.
    """
    global knn_model, vectorizer, respuestas_knn
    
    try:
        print("🔄 Cargando base de conocimiento FAQ desde Base de Datos...")
        query = "SELECT pregunta, respuesta FROM faq"
        df = pd.read_sql(query, engine)

        # Validación si la tabla está vacía
        if df.empty:
            print("⚠️ Advertencia: La tabla FAQ en la base de datos está vacía. El modelo KNN no sabrá nada.")
            return

        # Normalización de nombres de columnas (SQLAlchemy devuelve minúsculas, pero aseguramos)
        df.columns = [c.lower() for c in df.columns]

        # Validación de estructura
        if 'pregunta' not in df.columns or 'respuesta' not in df.columns:
            print("❌ Error KNN: La tabla FAQ no tiene las columnas 'pregunta' y 'respuesta'.")
            return

        # --- PROCESAMIENTO (Igual que antes) ---
        
        # Limpiamos las preguntas
        preguntas_limpias = [limpiar_texto(str(p)) for p in df['pregunta']]
        
        # Guardamos las respuestas originales en memoria
        respuestas_knn = df['respuesta'].tolist()

        # Entrenamiento
        vectorizer = CountVectorizer()
        X_dataset = vectorizer.fit_transform(preguntas_limpias)

        knn_model = NearestNeighbors(n_neighbors=1, metric='cosine')
        knn_model.fit(X_dataset)
        
        print(f"✅ Modelo KNN actualizado desde DB. Total conocimientos cargados: {len(respuestas_knn)}")

    except Exception as e:
       print(f"⚠️ Advertencia KNN: No se pudo cargar la base de datos (Puede que esté vacía o no exista la tabla). Usando solo LLM. Detalle: {e}")
    return
    # Aquí podrías poner un fallback si quisieras, pero mejor que falle para que te des cuenta.

# Carga inicial al importar el módulo
inicializar_knn()

def obtener_respuesta_knn(pregunta_usuario):
    """
    Recibe una pregunta, busca en el modelo KNN y retorna la respuesta más cercana
    junto con la distancia (similitud).
    """
    global knn_model, vectorizer, respuestas_knn
    
    # Si el modelo no cargó bien por error de DB, retornamos "lejos" para que use el LLM
    if not knn_model:
        return None, 1.0

    try:
        pregunta_usuario_limpia = limpiar_texto(pregunta_usuario)
        
        # Transformar input del usuario a vector
        X_usuario = vectorizer.transform([pregunta_usuario_limpia])
        
        # Buscar el vecino más cercano
        distancias, indices = knn_model.kneighbors(X_usuario)

        indice_respuesta = indices[0][0]
        distancia = distancias[0][0]
        
        respuesta = respuestas_knn[indice_respuesta]
        
        return respuesta, distancia

    except Exception as e:
        print(f"Error en predicción KNN: {e}")
        return None, 1.0