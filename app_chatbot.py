from flask import Blueprint, render_template, request, jsonify
import sys
import os

# --- INICIO DE LA LÓGICA DEL CHATBOT ---
# (Importamos la lógica de tu 'main_chatbot.py')

# Añadir la ruta al sys.path para encontrar los módulos de 'modelo'
# Esto asume que 'modelo' está en el directorio raíz, al mismo nivel que 'app.py'
# Si 'modelo' está en otro lugar, ajusta esta ruta.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Asumimos que la carpeta raíz está un nivel arriba de donde está este script (mi-chatbot-flask/app_chatbot.py)
# Si 'app_chatbot.py' está en la raíz (junto a 'app.py'), puedes quitar la siguiente línea:
# root_dir = os.path.dirname(script_dir) 
root_dir = script_dir # Si 'app_chatbot.py' está en la raíz
# Asumamos que 'modelo' está en la raíz
sys.path.append(root_dir)


from modelo.seleccion_modelo import SelectorDeModelo

# --- CONFIGURACIÓN ---
UMBRAL_KNN = 0.4 
selector_global = None

try:
    # Inicializa el selector UNA SOLA VEZ cuando se inicia la app.
    print("🤖 Iniciando backend del chatbot...")
    selector_global = SelectorDeModelo(
        usar_knn=True, 
        usar_llm=True, 
        umbral_distancia=UMBRAL_KNN
    )
    print("\n✅ Backend del Chatbot listo.")

except Exception as e:
    print("\n❌ ERROR CRÍTICO AL INICIAR EL BACKEND DEL CHATBOT ❌")
    print(f"No se pudo cargar el selector de modelos. El error fue:\n")
    print(f"   ➡️  {e}\n")
    print("Verifica las rutas a 'chroma_db_web' y 'faq.csv' y que Ollama esté corriendo.")

# --- FIN DE LA LÓGICA DEL CHATBOT ---


# Crear el Blueprint
chatbot_bp = Blueprint('chatbot', __name__)

@chatbot_bp.route('/chat')
def chat():
    """Sirve la página HTML principal del chatbot."""
    return render_template('chatbot.html', active_page='chat')


@chatbot_bp.route('/api/chat', methods=['POST'])
def api_chat():
    """Punto de entrada de la API para recibir mensajes y devolver respuestas."""
    
    if not selector_global:
        # Si el selector no se pudo cargar, devuelve un error
        return jsonify({
            "error": "El servicio de chatbot no está disponible."
        }), 500

    try:
        # Obtener el mensaje del JSON enviado por el frontend
        data = request.json
        pregunta = data.get('message')

        if not pregunta:
            return jsonify({"error": "No se recibió ningún mensaje."}), 400

        # Obtener la respuesta del selector (la lógica principal)
        respuesta, modelo_usado = selector_global.responder(pregunta)

        # Devolver la respuesta al frontend
        return jsonify({
            "reply": respuesta,
            "model": modelo_usado
        })

    except Exception as e:
        print(f"Error procesando la solicitud de chat: {e}")
        return jsonify({
            "error": "Ocurrió un error al procesar tu respuesta."
        }), 500