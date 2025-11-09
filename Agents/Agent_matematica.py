# =======================================================================
# Agents/Agent_matematica.py - Agente Especialista en Matemáticas
# =======================================================================

from typing import Any, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# =========================================
# 0. Inicialización LLM y memoria
# =========================================
llm = ChatOpenAI(temperature=0.4, model="gpt-4o-mini")
memory = MemorySaver()
tavily_tool = TavilySearchResults(max_results=4)

# =========================================
# 1. Schema de salida
# =========================================
class RespuestaMatematica(BaseModel):
    explicacion_profunda: str = Field(description="Explicación detallada del concepto, procedimiento o verificación.")
    parrafo_ejemplo: str = Field(description="Ejemplo práctico o problema resuelto que ilustra la explicación.")

# =========================================
# 2. Herramientas Matemáticas
# =========================================
@tool
def resolucion_problemas(problema: str) -> str:
    """Resuelve problemas matemáticos paso a paso."""
    system = SystemMessage(content=(
        "Eres un asistente de matemáticas para secundaria. "
        "Resuelve el problema paso a paso mostrando cálculos y concluye con la respuesta final. "
        "Indica cómo verificar la solución si aplica."
    ))
    human = HumanMessage(content=problema)
    resp = llm.invoke([system, human])
    return resp.content.strip()

@tool
def explicacion_concepto(concepto: str) -> str:
    """Explica conceptos matemáticos con ejemplos."""
    # Intentamos obtener contexto de Tavily
    contexto_text = ""
    try:
        raw_results = tavily_tool.invoke({"query": f"Definición y ejemplos: {concepto} matemáticas secundaria"})
        if isinstance(raw_results, list):
            contexto_text = "\n".join([r.get("content", "") for r in raw_results if isinstance(r, dict)])
        else:
            contexto_text = str(raw_results)
    except Exception as e:
        contexto_text = f"(No se pudo obtener contexto: {e})"

    system = SystemMessage(content=(
        f"Eres un profesor de matemáticas para secundaria. Usa el contexto cuando sea útil:\n{contexto_text}\n"
        "Explica el concepto claramente e incluye un ejemplo breve."
    ))
    human = HumanMessage(content=concepto)
    resp = llm.invoke([system, human])
    return resp.content.strip()

@tool
def verificacion_resultado(enunciado: str, respuesta_alumno: str) -> str:
    """Verifica la coherencia de la respuesta de un alumno y da retroalimentación."""
    system = SystemMessage(content=(
        "Eres un verificador pedagógico en matemáticas. "
        "Revisa el enunciado y la respuesta del alumno. "
        "Indica si es correcta, explica por qué o por qué no, y sugiere pasos de corrección."
    ))
    human = HumanMessage(content=f"Enunciado: {enunciado}\nRespuesta del alumno: {respuesta_alumno}")
    resp = llm.invoke([system, human])
    return resp.content.strip()

tools = [resolucion_problemas, explicacion_concepto, verificacion_resultado]

# =========================================
# 3. Prompt general para el agente
# =========================================
PROMPT_GENERAL = f"""
Eres EVA, un experto en Matemáticas para estudiantes de secundaria. 
Tu tarea es analizar la solicitud del usuario y decidir cuál herramienta usar:

- Si el usuario pide resolver un problema paso a paso, usa la herramienta **resolucion_problemas**.
- Si el usuario pide una explicación de un concepto matemático, usa la herramienta **explicacion_concepto**.
- Si el usuario pide verificar o corregir una respuesta de alumno, usa la herramienta **verificacion_resultado**.

Responde SIEMPRE en formato JSON compatible con Pydantic:
{{
  "explicacion_profunda": "Explicación detallada del concepto, procedimiento o verificación.",
  "parrafo_ejemplo": "Ejemplo práctico o problema resuelto que ilustra la explicación."
}}
"""

# =========================================
# 4. Crear agente ReAct
# =========================================
agent = create_react_agent(llm, tools, checkpointer=memory, prompt=PROMPT_GENERAL)

# =========================================
# 5. Función para Streamlit
# =========================================
global_llm_with_tools = None

def get_matematica_agent():
    """Inicializa y devuelve el agente de Matemáticas y su esquema Pydantic."""
    global global_llm_with_tools
    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente Matemáticas...")
        global_llm_with_tools = agent
        print("✅ Agente Matemáticas inicializado correctamente.")

    schema = {
        "explicacion_profunda": "str",
        "parrafo_ejemplo": "str"
    }
    return global_llm_with_tools, schema
