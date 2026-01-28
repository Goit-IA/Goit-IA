from flask import Blueprint, render_template, request, jsonify, redirect, url_for
import sys
import os
import re

# Configuración de rutas
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
template_dir = os.path.join(project_root, 'templates')
logic_dir = os.path.join(project_root, 'logic') 
models_dir = os.path.join(project_root, 'models')
data_dir_chroma = os.path.join(project_root, 'data', 'chroma_db_web')

# Aseguramos que la raíz del proyecto esté en el path para importar 'database.py'
if project_root not in sys.path: sys.path.append(project_root)

# --- IMPORTS DE BASE DE DATOS ---
from database import SessionLocal, FAQ

# Imports de lógica
from models import modelo_knn 
from models import modelo_llm
modelo_llm.CHROMA_PATH = data_dir_chroma
from logic.seleccion_modelo import SelectorDeModelo

# Import para el registro de accesos
from logic.access_tracker import registrar_acceso

# Inicialización del Blueprint
chatbot_bp = Blueprint('chatbot', __name__, template_folder=template_dir)

# Inicialización del Selector
selector = None
try:
    selector = SelectorDeModelo(usar_knn=True, usar_llm=True)
except Exception as e:
    print(f"Error al iniciar selector: {e}")

# --- FUNCIÓN PARA GUARDAR EN BASE DE DATOS ---
# --- EN app_chatbot.py ---

def guardar_faq_db(pregunta, respuesta):
    """
    Busca si la pregunta ya existe.
    - Si EXISTE: Actualiza la respuesta con la nueva versión (ideal para 'Regenerar').
    - Si NO EXISTE: Crea un registro nuevo.
    Esto asegura que no haya preguntas repetidas y siempre se tenga la última versión.
    """
    db = SessionLocal()
    try:
        # 1. Buscamos si ya existe la PREGUNTA (independientemente de la respuesta)
        registro_existente = db.query(FAQ).filter(FAQ.pregunta == pregunta).first()

        if registro_existente:
            # --- CASO: ACTUALIZAR ---
            print(f"🔄 Pregunta existente encontrada. Actualizando respuesta...")
            registro_existente.respuesta = respuesta
            # Nota: SQLAlchemy detecta el cambio y lo aplicará al hacer commit
        else:
            # --- CASO: CREAR NUEVO ---
            print(f"✅ Nueva pregunta detectada. Guardando...")
            nueva_faq = FAQ(pregunta=pregunta, respuesta=respuesta)
            db.add(nueva_faq)

        # 2. Confirmamos los cambios en la DB
        db.commit()
        
        # 3. Recargamos el modelo KNN para que aprenda el cambio inmediatamente
        try:
            modelo_knn.inicializar_knn()
        except Exception as e:
            print(f"⚠️ Error recargando KNN: {e}")

    except Exception as e:
        print(f"❌ Error en DB: {e}")
        db.rollback()
    finally:
        db.close()

# --- RUTA PRINCIPAL (VISTA) ---
@chatbot_bp.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

# --- RUTA CHAT ACTUALIZADA ---
@chatbot_bp.route('/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'GET':
        return redirect(url_for('chatbot.chatbot'))

    data = request.json
    user_input = data.get("message")
    mode = data.get("mode", "normal")
    
    if not user_input:
        return jsonify({"reply": "Por favor escribe algo."})

    # Si es 'regenerate', forzamos LLM. Si no, dejamos que el selector decida.
    forzar_llm = (mode == 'regenerate')

    # 1. Obtener respuesta
    respuesta_limpia, fuente = selector.responder(user_input, forzar_llm=forzar_llm)

    # 2. Lógica de Aprendizaje
    # AHORA: Guardamos siempre que venga del LLM, sin importar si es regeneración o no.
    # La función 'guardar_faq_db' se encarga de evitar duplicados.
    if "LLM" in fuente:
        pregunta_usuario = user_input.strip()
        guardar_faq_db(pregunta_usuario, respuesta_limpia)

    return jsonify({
        "reply": respuesta_limpia,
        "model": fuente
    })



# --- RUTA: REGISTRO DE ACCESOS ---
@chatbot_bp.route('/api/register_access', methods=['POST'])
def register_access():
    data = request.json
    programa = data.get('programa')
    
    if not programa:
        return jsonify({"status": "error", "message": "Programa no seleccionado"}), 400

    # Obtener IP (Manejo de Proxy si existe, sino remote_addr)
    if request.headers.getlist("X-Forwarded-For"):
        user_ip = request.headers.getlist("X-Forwarded-For")[0]
    else:
        user_ip = request.remote_addr
        
    user_agent = request.headers.get('User-Agent')

    # Guardar usando la lógica de access_tracker
    try:
        registrar_acceso(programa, user_ip, user_agent)
        return jsonify({"status": "success", "message": "Access logged"})
    except Exception as e:
        print(f"Error logging access: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500