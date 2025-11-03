# Asumiendo que los archivos están en una carpeta 'modelo'
# Si están en el mismo directorio, cambia las importaciones
try:
    from .modelo_knn import obtener_respuesta_knn
    from .modelo_llm import get_rag_chain
except ImportError:
    # Si se ejecutan desde el mismo directorio
    from modelo_knn import obtener_respuesta_knn
    from modelo_llm import get_rag_chain

class SelectorDeModelo:
    def __init__(self, usar_knn=True, usar_llm=True, umbral_distancia=0.4):
        """
        Inicializa el selector de modelos.
        
        :param usar_knn: Booleano para activar/desactivar el modelo KNN.
        :param usar_llm: Booleano para activar/desactivar el modelo LLM.
        :param umbral_distancia: Flotante. La distancia de coseno máxima
                                 para considerar una coincidencia de KNN.
                                 Un valor más bajo es MÁS estricto.
                                 (0.0 = coincidencia exacta, 1.0 = opuesto total)
        """
        self.usar_knn = usar_knn
        self.usar_llm = usar_llm
        
        # --- VARIABLE CONFIGURABLE ---
        self.UMBRAL_DISTANCIA_COSINE = umbral_distancia
        # -----------------------------

        self.rag_chain = None
        
        # Inicializa el modelo LLM (RAG) si está habilitado
        # Esto puede tardar un momento, por eso se hace en el __init__
        if self.usar_llm:
            try:
                print("Iniciando y cargando el modelo LLM (RAG)...")
                # Obtenemos la cadena RAG completa de modelo_llm
                self.rag_chain = get_rag_chain()
                print("✅ Modelo LLM listo.")
            except Exception as e:
                print(f"❌ ERROR CRÍTICO al inicializar el modelo LLM: {e}")
                print("El modo LLM se ha desactivado.")
                self.usar_llm = False

    def responder(self, pregunta):
        """
        Genera una respuesta usando KNN con fallback a LLM basado en el umbral.
        Devuelve (respuesta, modelo_utilizado)
        """
        
        # --- PASO 1: Intentar con KNN ---
        if self.usar_knn:
            respuesta_knn, distancia = obtener_respuesta_knn(pregunta)
            
            # Imprime para depuración (opcional)
            print(f"DEBUG: Distancia KNN = {distancia:.4f} (Umbral: {self.UMBRAL_DISTANCIA_COSINE})")

            # Si la respuesta es válida y la distancia está DENTRO del umbral
            if respuesta_knn and distancia <= self.UMBRAL_DISTANCIA_COSINE:
                return respuesta_knn, "KNN (Coincidencia Alta)"

        # --- PASO 2: Fallback a LLM ---
        # Si KNN está desactivado, o no encontró respuesta, o la distancia fue muy alta
        if self.usar_llm:
            if not self.rag_chain:
                return "Error: El modelo LLM se activó pero no se inicializó correctamente.", "Error"
            
            # El modelo LLM (RAG) genera la respuesta
            try:
                respuesta_llm = self.rag_chain.invoke(pregunta)
                return respuesta_llm, "LLM (RAG)"
            except Exception as e:
                print(f"Error al invocar la cadena RAG: {e}")
                return "Lo siento, ocurrió un error al procesar tu pregunta con el LLM.", "Error LLM"

        # --- PASO 3: Si ambos fallan o están desactivados ---
        return "Lo siento, no tengo una respuesta disponible para esa pregunta.", "Sin Modelo"