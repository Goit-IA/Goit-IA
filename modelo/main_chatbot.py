import sys
import threading

# Asumiendo que los archivos están en una carpeta 'modelo'
# Si están en el mismo directorio, cambia las importaciones
try:
    from modelo.seleccion_modelo import SelectorDeModelo
    from modelo.modelo_llm import mostrar_barra_de_carga
except ImportError:
    # Si se ejecutan desde el mismo directorio
    from seleccion_modelo import SelectorDeModelo
    from modelo_llm import mostrar_barra_de_carga

def main():
    """
    Función principal que inicia el chatbot interactivo
    utilizando el SelectorDeModelo.
    """
    print("🤖 Iniciando chatbot con selector de modelos...")
    
    # --- CONFIGURACIÓN ---
    # Aquí puedes ajustar el umbral de similitud para KNN.
    # Un valor más bajo (ej. 0.2) significa que debe ser MUY similar
    # para usar KNN.
    # Un valor más alto (ej. 0.6) permite más flexibilidad a KNN.
    UMBRAL_KNN = 0.4 
    # ---------------------

    try:
        # Inicializa el selector. Esto cargará el modelo LLM.
        selector = SelectorDeModelo(
            usar_knn=True, 
            usar_llm=False, 
            umbral_distancia=UMBRAL_KNN
        )
        print("\n✅ Chatbot listo. Pregúntame sobre los trámites de la UV. Escribe 'salir' para terminar.")
        print("-" * 70)
    
    except Exception as e:
        # Captura cualquier error durante la inicialización (ej. Chroma no encontrado)
        print("\n❌ ERROR CRÍTICO AL INICIAR EL CHATBOT ❌")
        print(f"No se pudo cargar el selector de modelos. El error fue:\n")
        print(f"   ➡️  {e}\n")
        print("POSIBLES CAUSAS:")
        print("   1. La carpeta de la base de datos Chroma ('chroma_db_web') no existe.")
        print("   2. El modelo de Ollama ('phi3:mini' o 'nomic-embed-text') no está disponible.")
        print("   3. El archivo '../faq.csv' no se encontró (para el modelo KNN).")
        sys.exit(1)

    # Bucle de chat principal
    while True:
        pregunta = input("Tú: ")
        
        if pregunta.lower() == 'salir':
            print("\n🤖 ¡Hasta luego! Ha sido un placer ayudarte.")
            break
        
        evento_parada = threading.Event()
        resultado = {"respuesta": None, "modelo": None, "error": None}

        def obtener_respuesta_wrapper(p):
            """Función wrapper para ejecutar en el hilo."""
            try:
                # El selector decide qué modelo usar
                respuesta, modelo_usado = selector.responder(p)
                resultado["respuesta"] = respuesta
                resultado["modelo"] = modelo_usado
            except Exception as e:
                resultado["error"] = e
            finally:
                evento_parada.set()

        hilo_trabajo = threading.Thread(target=obtener_respuesta_wrapper, args=(pregunta,))
        hilo_trabajo.start()

        # Muestra la animación de carga mientras el hilo trabaja
        mostrar_barra_de_carga(evento_parada)

        hilo_trabajo.join()

        # Muestra el resultado
        if resultado["error"]:
            print(f"\nLo siento, ocurrió un error al procesar tu pregunta.")
            print(f"Detalle del error: {resultado['error']}")
        else:
            # Imprime la respuesta e incluye qué modelo la generó
            print(f"Chatbot (vía {resultado['modelo']}): {resultado['respuesta']}")
        
        print("-" * 70)


if __name__ == "__main__":
    main()