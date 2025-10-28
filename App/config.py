# app/config.py
import os
import sys

# =======================================================================
# 1. FUNCIÓN DE UTILIDAD: Carga de Claves desde LOGS
# =======================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "..", "logs")

def _load_api_key(filename):
    """Lee y devuelve una clave API de un archivo en la carpeta LOGS."""
    file_path = os.path.join(LOGS_DIR, filename)
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Archivo de clave '{filename}' no encontrado en la ruta: {file_path}")
    except Exception as e:
        raise Exception(f"Error al leer la clave en {filename}: {e}")

# =======================================================================
# 2. FUNCIÓN PRINCIPAL: Carga de Configuración de Entorno
# =======================================================================

def load_config_and_keys():
    """
    Carga todas las claves API y configura las variables de entorno
    requeridas para OpenAI, LangSmith y Tavily.
    """
    print("🚀 Cargando configuración de entorno...")

    try:
        # 🔑 OpenAI (LLM)
        os.environ["OPENAI_API_KEY"] = _load_api_key("clave_api.txt")
        print("   ✅ Clave OpenAI cargada.")

        # 📊 LangSmith (Trazabilidad y Observabilidad)
        os.environ["LANGSMITH_API_KEY"] = _load_api_key("langgraphapi.txt")
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "EVA_Project_Tracing"
        print("   ✅ Configuración de LangSmith cargada.")

        # 🌐 Tavily (Búsqueda contextual)
        os.environ["TAVILY_API_KEY"] = _load_api_key("tavily_api.txt")
        print("   ✅ Clave Tavily cargada.")

        # 🔧 Aquí puedes añadir futuras integraciones (ej. HuggingFace, SerpAPI, etc.)

    except FileNotFoundError as e:
        print(f"🛑 ERROR CRÍTICO: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"🛑 ERROR CRÍTICO durante la carga de configuración: {e}")
        sys.exit(1)
