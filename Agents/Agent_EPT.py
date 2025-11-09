# Agents/Agent_ept.py
import json
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults

# =========================================
# LLM Y MEMORIA
# =========================================
llm = ChatOpenAI(temperature=0.4, model="gpt-4o-mini")
memory = MemorySaver()

# =========================================
# TOOLS DEFINIDAS (EPT)
# =========================================

# 1) Planificación de proyectos educativos
@tool
def plan_proyecto(tema: str) -> str:
    """
    Genera la estructura completa de un proyecto educativo sobre un tema dado.
    Incluye objetivos, materiales, pasos y evaluación.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.25)
    system = SystemMessage(content=(
        "Eres un docente de Educación para el Trabajo (EPT). "
        "Estructura un proyecto educativo claro con objetivos, materiales, pasos y evaluación."
    ))
    human = HumanMessage(content=f"Tema del proyecto: {tema}")
    resp = llm.invoke([system, human])
    return resp.content.strip()


# 2) Explicación de conceptos tecnológicos
@tool
def concepto_tecnologico(concepto: str) -> str:
    """
    Explica un concepto o herramienta tecnológica de forma clara y concisa,
    incluyendo su aplicación práctica en proyectos educativos.
    """
    contexto_text = ""
    try:
        tavily = TavilySearchResults(max_results=3)
        raw_results = tavily.invoke({"query": f"Concepto tecnológico educativo: {concepto}"})
        if isinstance(raw_results, list):
            contexto_text = "\n".join([r.get("content", "") for r in raw_results if isinstance(r, dict)])
    except Exception as e:
        contexto_text = f"(No se pudo obtener contexto de Tavily: {e})"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    system = SystemMessage(content=(
        "Eres un profesor de EPT especializado en tecnología. "
        "Explica el concepto de forma pedagógica y añade un ejemplo práctico simple."
    ))
    human = HumanMessage(content=f"Concepto: {concepto}\n\nContexto:\n{contexto_text}")
    resp = llm.invoke([system, human])
    return resp.content.strip()


# 3) Evaluación de proyectos
@tool
def evaluacion_proyecto(descripcion: str) -> str:
    """
    Evalúa la viabilidad pedagógica de un proyecto educativo.
    Sugiere mejoras en objetivos, metodología o recursos.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(content=(
        "Eres un especialista pedagógico en evaluación de proyectos de EPT. "
        "Analiza la viabilidad del proyecto y da sugerencias claras de mejora."
    ))
    human = HumanMessage(content=f"Descripción del proyecto:\n{descripcion}")
    resp = llm.invoke([system, human])
    return resp.content.strip()


# Lista de herramientas
tools = [plan_proyecto, concepto_tecnologico, evaluacion_proyecto]

# =========================================
# PROMPT BASE REACT
# =========================================
prompt = """
Eres EVA, una especialista en Educación para el Trabajo (EPT).
Tu tarea es analizar la solicitud del usuario y decidir qué herramienta usar.

- Si el usuario pide estructurar o planificar un proyecto, usa la herramienta **plan_proyecto**.
- Si el usuario pide la definición o explicación de un concepto tecnológico, usa **concepto_tecnologico**.
- Si el usuario pide evaluar o mejorar un proyecto, usa **evaluacion_proyecto**.

Responde siempre en formato JSON con los siguientes campos:
{
  "explicacion_profunda": "explicación o desarrollo del tema solicitado",
  "parrafo_ejemplo": "ejemplo o aplicación práctica si aplica, o vacío si no aplica"
}
"""

# =========================================
# CREAR EL AGENTE REACT CON HERRAMIENTAS
# =========================================
agent = create_react_agent(llm, tools, checkpointer=memory, prompt=prompt)

# =========================================
# FUNCIÓN PARA STREAMLIT
# =========================================
global_llm_with_tools = None  # Inicializar variable global

def get_ept_agent():
    """Inicializa y devuelve el agente de Educación para el Trabajo (EPT) y su esquema."""
    global global_llm_with_tools

    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente EPT (LangGraph ReAct)...")
        global_llm_with_tools = agent
        print("✅ Agente EPT inicializado correctamente.")

    schema = {
        "explicacion_profunda": "str",
        "parrafo_ejemplo": "str"
    }

    return global_llm_with_tools, schema
