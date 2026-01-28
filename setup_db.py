# setup_db.py
from database import init_db

if __name__ == "__main__":
    print("🛠️ Iniciando configuración de base de datos...")
    try:
        init_db()
        print("🚀 Base de datos lista. Ahora puedes ejecutar app_chatbot.py")
    except Exception as e:
        print(f"❌ Error al crear tablas: {e}")