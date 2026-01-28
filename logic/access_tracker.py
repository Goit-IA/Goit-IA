import pandas as pd
from datetime import datetime
import sys
import os

# --- CONFIGURACIÓN DE RUTAS ---
# Obtenemos la ruta del directorio actual (logic)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Obtenemos la raíz del proyecto (un nivel arriba)
project_root = os.path.dirname(current_dir)

# Agregamos la raíz al path para poder importar 'database.py'
if project_root not in sys.path:
    sys.path.append(project_root)

# --- IMPORTS DE BASE DE DATOS ---
from database import SessionLocal, AccessLog, engine

def registrar_acceso(programa, ip, dispositivo):
    """
    Guarda el registro de acceso en la Base de Datos PostgreSQL.
    """
    ahora = datetime.now()
    
    # Abrimos una sesión temporal con la base de datos
    db = SessionLocal()
    
    try:
        # Creamos el objeto (fila) para insertar
        nuevo_log = AccessLog(
            dia=ahora.strftime('%A'),
            fecha=ahora.strftime('%Y-%m-%d'),
            hora=ahora.strftime('%H:%M:%S'),
            programa=programa,
            dispositivo=dispositivo,
            ip=ip
        )
        
        # Guardamos y confirmamos cambios
        db.add(nuevo_log)
        db.commit()
        print(f"✅ Acceso registrado en DB: {programa} desde {ip}")
        return True
        
    except Exception as e:
        print(f"❌ Error registrando acceso en DB: {e}")
        db.rollback() # Deshacemos cambios si hubo error
        return False
        
    finally:
        db.close() # Cerramos la conexión siempre

def obtener_estadisticas_diarias():
    """
    Devuelve el conteo de visitas por programa para las gráficas.
    Lee directamente desde PostgreSQL usando Pandas.
    """
    try:
        # Consulta SQL optimizada: solo traemos la columna que nos interesa contar
        query = "SELECT programa FROM access_log"
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return {}
            
        # Contamos cuántas veces aparece cada programa
        return df['programa'].value_counts().to_dict()
        
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")
        return {}

def obtener_todos_los_registros():
    """
    Devuelve todos los registros para la tabla del panel de administración.
    Ordenados del más reciente al más antiguo.
    """
    try:
        # Consulta SQL ordenando por fecha y hora descendente
        query = "SELECT * FROM access_log ORDER BY fecha DESC, hora DESC"
        
        df = pd.read_sql(query, engine)
        
        if df.empty:
            return []
            
        # Convertimos a lista de diccionarios para que Jinja (HTML) lo renderice igual que antes
        return df.to_dict(orient='records')
        
    except Exception as e:
        print(f"❌ Error obteniendo registros históricos: {e}")
        return []