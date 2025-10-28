# test_validator.py - VERSIÓN MINIMALISTA

import os
import sys
import json
from typing import Dict

# ---------------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS Y CLAVES
# ---------------------------------------------------------------------

# 1. Obtenemos el path del directorio 'App' (donde está este archivo)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Subimos un nivel para encontrar el directorio principal del proyecto ('EVA')
project_root = os.path.dirname(current_dir)

# 3. Añadimos el directorio principal ('EVA') al sys.path.
if project_root not in sys.path:
    sys.path.append(project_root)


# -----------------------------------------------
# IMPORTACIONES
# -----------------------------------------------

# Cargar claves API
from App.config import load_config_and_keys
load_config_and_keys()

# Importar el módulo validador
import App.validador as validador

# ---------------------------------------------------------------------
# Función de Prueba (SIMPLIFICADA)
# ---------------------------------------------------------------------

def test_luzia_pipeline_output_only(pregunta: str, curso_set: str, grado_set: str):
    """
    Ejecuta el pipeline de Luzia y SOLO imprime el diccionario de salida.
    """
    try:
        # Llamamos a la función de envoltura para la ejecución
        resultado: Dict = validador.run_luzia_pipeline( 
            grado_sistema=grado_set, 
            curso_sistema=curso_set, 
            pregunta=pregunta
        )
        
        # Imprime el diccionario de salida tal cual (como lo vería main.py)
        print(json.dumps(resultado, indent=4, ensure_ascii=False))

    except Exception as e:
        # Si falla, simplemente imprime el error.
        print(f"❌ FALLÓ LA EJECUCIÓN DEL PIPELINE. Error: {e}")

# ---------------------------------------------------------------------
# Ejecutar la prueba
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Caso de Prueba que falla (Maestría en Comunicación en 1° Secundaria)
    print("\n--- INICIO DE PRUEBA DE SALIDA (raw output) ---")
    test_luzia_pipeline_output_only(
        pregunta="que es la fotosintesis",
        curso_set="Ciencia y Tecnología",
        grado_set="1° Secundaria"
    )
    print("--- FIN DE PRUEBA ---")