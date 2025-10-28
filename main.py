# =======================================================================
# main.py - Orquestador Principal (Versión Unificada con todos los Agentes)
# =======================================================================
import os
import sys
import json
from pydantic import ValidationError
from langchain_core.messages import HumanMessage

# Añade los paths de módulos (App y Agents)
sys.path.append(os.path.join(os.path.dirname(__file__), "App"))
sys.path.append(os.path.join(os.path.dirname(__file__), "Agents"))

# 🚨 1. CARGA DE CONFIGURACIÓN Y CLAVES
from App.config import load_config_and_keys
load_config_and_keys()

# 🚨 2. IMPORTACIÓN DE VALIDADOR Y AGENTES
from App.validador import run_luzia_pipeline

from Agents.Agent_comunicacion import get_comunicacion_agent
from Agents.Agent_matematica import get_matematica_agent
from Agents.Agent_CTA import get_cta_agent
from Agents.Agent_EPT import get_ept_agent
from Agents.Agent_ingles import get_ingles_agent  # ← NUEVO AGENTE DE INGLÉS

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
    INGLES_EXECUTOR, _ = get_ingles_agent()  # Inicialización del agente Inglés

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
def procesar_pregunta(pregunta: str, grado_sistema: str, curso_sistema: str) -> str:
    """
    Ruta la pregunta a través del validador y luego invoca al agente especialista correspondiente.
    """
    print(f"Procesando Pregunta: Grado={grado_sistema}, Curso={curso_sistema}")

    try:
        resultado_validacion = run_luzia_pipeline(grado_sistema, curso_sistema, pregunta)
    except Exception as e:
        return f"❌ **Error Crítico del Sistema (API/LLM):** {type(e).__name__}: {e}"

    # Desempaquetar diagnóstico
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

    # Validación previa
    if not es_valido:
        return f"⚠️ **Advertencia del Validador:**\n\n{mensaje_diagnostico}"

    # Verificar si el curso tiene agente
    executor = AGENTS_EXECUTORS.get(curso_destino)
    if not executor:
        return f"❓ **Error de Ruteo:** No hay agente configurado para '{curso_destino}'."

    # Invocar agente
    try:
        respuesta_llm = executor.invoke(
            {"messages": [HumanMessage(content=prompt_para_agente)]},
            config={"configurable": {"thread_id": f"{curso_destino}_session_1"}},
        )

        # --- Limpieza y Formateo de salida ---
        if isinstance(respuesta_llm, dict) and "messages" in respuesta_llm:
            mensajes = respuesta_llm["messages"]
            respuesta_final = next(
                (
                    m.content.strip()
                    for m in reversed(mensajes)
                    if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip()
                ),
                None,
            )
        elif hasattr(respuesta_llm, "content"):
            respuesta_final = respuesta_llm.content.strip()
        else:
            respuesta_final = str(respuesta_llm).strip()

        if not respuesta_final:
            respuesta_final = "(Respuesta vacía del agente)"

        respuesta_final = respuesta_final.lstrip("json").strip()

        # Intentar parsear JSON
        data = None
        if respuesta_final.startswith("{") or respuesta_final.startswith("["):
            try:
                data = json.loads(respuesta_final)
            except json.JSONDecodeError:
                try:
                    respuesta_final_corr = (
                        respuesta_final.replace("'", '"')
                        .replace("\n", " ")
                        .replace("“", '"').replace("”", '"')
                    )
                    data = json.loads(respuesta_final_corr)
                except Exception:
                    data = None

        # Formateo final
        if data and isinstance(data, dict):
            explicacion = data.get("explicacion_profunda", "").strip()
            ejemplo = data.get("parrafo_ejemplo", "").strip()
            respuesta_final = ""
            if explicacion:
                respuesta_final += f"**Explicación:** {explicacion}\n\n"
            if ejemplo:
                respuesta_final += f"**Ejemplo:** {ejemplo}"
        else:
            respuesta_final = respuesta_final.replace("{", "").replace("}", "").replace('"', "").replace("'", "").strip()

        return f"✅ **Respuesta del Agente Especialista ({curso_destino}):**\n\n{respuesta_final}"

    except Exception as e:
        return f"❌ **Error en la Ejecución del Agente de {curso_destino}:**\n\n`{type(e).__name__}: {e}`"
