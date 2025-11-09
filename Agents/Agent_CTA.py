# Agents/Agent_CTA.py
## Imports
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
llm = ChatOpenAI(temperature=0.35, model="gpt-4o-mini")
memory = MemorySaver()

# =========================================
# HERRAMIENTAS (TOOLS)
# =========================================

# 1) Explicación científica → definición o descripción de fenómeno
@tool
def explicacion_cientifica(concepto: str) -> str:
    """
    Explica un fenómeno natural, proceso biológico o físico de forma clara, correcta y comprensible.
    No propone experimentos ni análisis, solo explicación teórica.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    system = SystemMessage(content=(
        "Eres un profesor de Ciencias, Tecnología y Ambiente. "
        "Explica de forma clara, rigurosa y comprensible conceptos científicos o procesos naturales. "
        "No generes ejemplos experimentales aquí."
    ))
    resp = llm.invoke([system, HumanMessage(content=f"Explica: {concepto}")])
    return resp.content.strip()


# 2) Experimento sugerido → híbrido Tavily + LLM
@tool
def experimento_sugerido(concepto: str) -> str:
    """
    Propone un experimento educativo o simulación sencilla para comprobar un fenómeno científico.
    Usa Tavily para buscar ideas o contextos experimentales y redacta una versión práctica y segura.
    """
    contexto_text = ""
    try:
        tavily = TavilySearchResults(max_results=4)
        raw_results = tavily.invoke({"query": f"Experimento educativo sobre {concepto}"})
        if isinstance(raw_results, list):
            contexto_text = "\n".join([r.get("content", "") for r in raw_results if isinstance(r, dict)])
        else:
            contexto_text = str(raw_results)
    except Exception as e:
        contexto_text = f"(No se pudo obtener contexto de Tavily: {e})"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.45)
    system = SystemMessage(content=(
        "Eres un profesor de CTA que sugiere experimentos seguros y didácticos para estudiantes de secundaria. "
        "Usa el CONTEXTO si es útil, pero describe solo un experimento breve y realista."
    ))
    human = HumanMessage(content=(
        f"CONTEXTO web:\n{contexto_text}\n\n"
        f"Propón un experimento sencillo para comprobar o demostrar: {concepto}"
    ))
    resp = llm.invoke([system, human])
    return resp.content.strip()


# 3) Análisis de impacto → reflexión sobre sostenibilidad
@tool
def analisis_impacto(tema: str) -> str:
    """
    Analiza los impactos ambientales o tecnológicos de un tema y propone soluciones sostenibles.
    Usa solo el LLM, sin búsqueda externa.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    system = SystemMessage(content=(
        "Eres un especialista en sostenibilidad y medio ambiente. "
        "Analiza de forma objetiva los efectos positivos y negativos del tema, "
        "y plantea una o dos soluciones prácticas sostenibles."
    ))
    resp = llm.invoke([system, HumanMessage(content=f"Analiza los impactos ambientales o tecnológicos de: {tema}")])
    return resp.content.strip()


# Lista de herramientas
tools = [explicacion_cientifica, experimento_sugerido, analisis_impacto]

# =========================================
# PROMPT BASE DEL AGENTE CTA
# =========================================
prompt = """
Eres EVA, una especialista en Ciencias, Tecnología y Ambiente (CTA).
Tu tarea es analizar la pregunta del usuario y decidir qué herramienta usar:

- Si el usuario pide una **explicación o definición** de un concepto o fenómeno, usa **explicacion_cientifica**.
- Si el usuario pide un **experimento o simulación**, usa **experimento_sugerido**.
- Si el usuario pide un **análisis de impacto ambiental o tecnológico**, usa **analisis_impacto**.

Responde siempre en formato JSON con los siguientes campos:
{
  "explicacion_profunda": "Explicación o análisis del fenómeno o tema",
  "parrafo_ejemplo": "Ejemplo, experimento o propuesta aplicada (vacío si no aplica)"
}
"""

# =========================================
# CREACIÓN DEL AGENTE REACT
# =========================================
agent = create_react_agent(llm, tools, checkpointer=memory, prompt=prompt)

# =========================================
# FUNCIÓN PARA STREAMLIT / ORQUESTADOR
# =========================================
global_llm_with_tools = None

def get_cta_agent():
    """Inicializa y devuelve el agente de CTA y su esquema."""
    global global_llm_with_tools

    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente CTA (LangGraph ReAct)...")
        global_llm_with_tools = agent
        print("✅ Agente CTA inicializado correctamente.")

    schema = {
        "explicacion_profunda": "str",
        "parrafo_ejemplo": "str"
    }

    return global_llm_with_tools, schema
