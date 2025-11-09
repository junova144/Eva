# =======================================================================
# main.py - Orquestador Principal (Versión Unificada con todos los Agentes)
# =======================================================================
import os
import json 
import sys
import json
from pydantic import ValidationError
from langchain_core.messages import HumanMessage

# Añade los paths de módulos (App y Agents)
sys.path.append(os.path.join(os.path.dirname(__file__), "App"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Agents"))

# 1. CARGA DE CONFIGURACIÓN Y CLAVES
from App.config import load_config_and_keys
load_config_and_keys()

# 2. IMPORTACIÓN DE VALIDADOR Y AGENTES
from App.validador import run_eva_pipeline

from Agents.Agent_comunicacion import get_comunicacion_agent
from Agents.Agent_matematica import get_matematica_agent
from Agents.Agent_CTA import get_cta_agent
from Agents.Agent_EPT import get_ept_agent
from Agents.Agent_ingles import get_ingles_agent  

# -----------------------------------------------------------------------
# INICIALIZACIÓN GLOBAL: Carga y compilación de agentes
# -----------------------------------------------------------------------
AGENTS_EXECUTORS = {}

try:
    print("--- Inicializando Orquestador de Agentes ---")
    COMUNICACION_EXECUTOR, _ = get_comunicacion_agent()
    MATEMATICA_EXECUTOR, _ = get_matematica_agent()
    CTA_EXECUTOR, _ = get_cta_agent()
    EPT_EXECUTOR, _ = get_ept_agent()
    INGLES_EXECUTOR, _ = get_ingles_agent()  

    AGENTS_EXECUTORS.update({
        "Comunicación": COMUNICACION_EXECUTOR,
        "Matemática": MATEMATICA_EXECUTOR,
        "Ciencia y Tecnología": CTA_EXECUTOR,
        "Educación para el Trabajo": EPT_EXECUTOR,
        "Inglés": INGLES_EXECUTOR  # Añadido al diccionario
    })
    print("--- Todos los agentes inicializados ✅ ---")
except Exception as e:
    print(f"❌ ERROR al inicializar Agentes: {e}")

# =======================================================================
# 3. FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# =======================================================================
     # Activación del Flujo y Control de Fallos Críticos (API/LLM)
def procesar_pregunta(pregunta: str, grado_sistema: str, curso_sistema: str) -> str:
    """
    Ruta la pregunta a través del validador y luego invoca al agente especialista correspondiente.
    """
    print(f"Procesando Pregunta: Grado={grado_sistema}, Curso={curso_sistema}")

    try:
        resultado_validacion = run_eva_pipeline(grado_sistema, curso_sistema, pregunta)
    except Exception as e:
        return f"❌ **Error Crítico del Sistema (API/LLM):** {type(e).__name__}: {e}"

    # Desempaquetado del Diagnóstico y Control de Formato JSON
    try:
        diagnostico_json_str = resultado_validacion.get(
            "validacion_json", '{"valido": false, "mensaje": "Error interno."}'
        )
        diagnostico = json.loads(diagnostico_json_str)
        es_valido = diagnostico.get("valido", False)
        mensaje_diagnostico = diagnostico.get("mensaje", "")
        prompt_para_agente = resultado_validacion.get("prompt_final", "")
        curso_destino = resultado_validacion.get("curso_final", curso_sistema)
    except json.JSONDecodeError:
        return "❌ **Error de Parseo:** JSON mal formado desde el validador."


    # Bloqueo Lógico y Retorno Anticipado (si el validador es false)
    if not es_valido:
        valor_limpio = mensaje_diagnostico.strip().lstrip('{ "').rstrip('}" ').split(":", 1)[1].strip()
        mensaje_dict = {"respuesta": valor_limpio}
        return f"⚠️ **Advertencia del Validador:**\n\n{mensaje_dict['respuesta']}"


    # Verificar si el curso tiene agente
    executor = AGENTS_EXECUTORS.get(curso_destino) #validador decidio el curso y filtra al agente
    if not executor:
        return f"❓ **Error de Ruteo:** No hay agente configurado para '{curso_destino}'."

    # Invocar agente
    try:
        respuesta_llm = executor.invoke(
            {"messages": [HumanMessage(content=prompt_para_agente)]},
            config={"configurable": {"thread_id": f"{curso_destino}_session_1"}},
        )
        
        # --- Limpieza y Formateo de salida ---
        respuesta_final = None

        # Extraer contenido principal del resultado del LLM
        if hasattr(respuesta_llm, "content") and respuesta_llm.content:
            respuesta_final = respuesta_llm.content.strip()
        elif isinstance(respuesta_llm, dict) and "messages" in respuesta_llm:
            mensajes = respuesta_llm["messages"]
            for m in reversed(mensajes):
                if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip():
                    respuesta_final = m.content.strip()
                    break

        if not respuesta_final:
            return f"⚠️ El agente de {curso_destino} no devolvió contenido útil."

        # Quitar posibles etiquetas o formato erróneo
        respuesta_final = respuesta_final.replace("```json", "").replace("```", "").strip()

        # Intentar parsear JSON (estructura estándar de tus agentes)
        try:
            data = json.loads(respuesta_final)
        except Exception:
            data = None

        if isinstance(data, dict):
            explicacion = data.get("explicacion_profunda", "").strip()
            ejemplo = data.get("parrafo_ejemplo", "").strip()

            salida = f"✅ **Respuesta del Agente Especialista ({curso_destino}):**\n\n"
            if explicacion:
                salida += f"🧩 **Explicación:**\n{explicacion}\n\n"
            if ejemplo:
                salida += f"✏️ **Ejemplo:**\n{ejemplo}"
            return salida
        else:
            return f"✅ **Respuesta del Agente Especialista ({curso_destino}):**\n\n{respuesta_final}"

    except Exception as e:
        return f"❌ **Error en la Ejecución del Agente de {curso_destino}:**\n\n`{type(e).__name__}: {e}`"


##if __name__ == "__main__":
##    print("🧠 Iniciando prueba del agente Comunicación...")
##    try:
##        respuesta = procesar_pregunta(
##            "Qué es un texto argumentativo",
##            "1° Secundaria",
##            "matemática"
##        )
##        print("\n✅ DEBUG RESPUESTA AGENTE:")
##        print(respuesta)
##    except Exception as e:
##        print("\n❌ Error durante la prueba del agente:")
##        print(e)