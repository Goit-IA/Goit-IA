# database.py
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv # Importante cargar esto

load_dotenv() # Cargar variables del .env

# --- CONFIGURACIÓN ---
# Opción A: Usar variables de entorno (Recomendado)
# Asegúrate de tener DATABASE_URL en tu archivo .env
# Ejemplo en .env: DATABASE_URL="postgresql://postgres:tu_password@localhost:5432/goit_local"
DATABASE_URL = os.getenv("DATABASE_URL")

# Opción B (Si prefieres dejarlo fijo por ahora):
# Cambia 'tu_password' por TU contraseña real de Postgres
# DATABASE_URL = "postgresql://postgres:tu_password_real@localhost:5432/goit_local"

if not DATABASE_URL:
    raise ValueError("❌ Error: No se ha definido DATABASE_URL en el archivo .env o en el código.")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELOS (TABLAS) ---
class FAQ(Base):
    __tablename__ = "faq"
    id = Column(Integer, primary_key=True, index=True)
    pregunta = Column(Text, nullable=False)
    respuesta = Column(Text, nullable=False)

class AccessLog(Base):
    __tablename__ = "access_log"
    id = Column(Integer, primary_key=True, index=True)
    dia = Column(String(50))
    fecha = Column(String(50))
    hora = Column(String(50))
    programa = Column(String(100))
    dispositivo = Column(Text)
    ip = Column(String(50))

def init_db():
    """Crea las tablas en la base de datos"""
    print("🔄 Creando tablas en la base de datos...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tablas creadas exitosamente.")