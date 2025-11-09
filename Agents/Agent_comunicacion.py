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
llm = ChatOpenAI(temperature=0.4, model="gpt-4o-mini")
memory = MemorySaver()

# =========================================
# TOOLS DEFINIDAS
# =========================================

# 1) Comprensión de definiciones → solo LLM
@tool
def comprension_texto(texto: str) -> str:
    """
    Explica o define un concepto o tipo de texto de forma clara y concisa.
    No genera ejemplos ni corrige textos.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.15)
    system = SystemMessage(content=(
        "Eres un especialista en comunicación y lenguaje. Da definiciones claras y concisas, "
        "pensadas para estudiantes de secundaria. Si la pregunta es breve, responde con una definición corta. "
        "No añadas ejemplos ni formato JSON aquí — esta herramienta solo devuelve texto plano."
    ))
    resp = llm.invoke([system, HumanMessage(content=texto)])
    return resp.content.strip()


# 2) Producción de ejemplos → híbrido Tavily + LLM
@tool
def produccion_texto(tema_o_tipo_texto: str) -> str:
    """
    SOLO genera ejemplos o párrafos aplicados (nunca definiciones ni explicaciones teóricas).
    Usa Tavily para obtener contexto y redacta un ejemplo educativo práctico 
    para estudiantes de secundaria.
    """
    contexto_text = ""
    try:
        tavily = TavilySearchResults(max_results=4)
        raw_results = tavily.invoke({"query": f"Ejemplo educativo: {tema_o_tipo_texto}"})
        if isinstance(raw_results, list):
            contexto_text = "\n".join(
                [r.get("content", "") for r in raw_results if isinstance(r, dict)]
            )
        else:
            contexto_text = str(raw_results)
    except Exception as e:
        contexto_text = f"(No se pudo obtener contexto de Tavily: {e})"

    # Modelo con ligera creatividad
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.45)

    # Reforzamos el rol y el límite del tipo de salida
    system = SystemMessage(content=(
        "Eres un redactor educativo especializado en crear ejemplos prácticos. "
        "Tu tarea es redactar solo ejemplos o párrafos aplicados. "
        "Nunca des definiciones ni explicaciones teóricas. "
        "Usa el CONTEXTO si es útil, pero redacta un ejemplo claro y natural para estudiantes de secundaria."
    ))

    # Prompt explícito sobre qué producir
    human = HumanMessage(content=(
        f"Tema o tipo de texto: {tema_o_tipo_texto}\n\n"
        f"CONTEXTO web relevante:\n{contexto_text}\n\n"
        "Genera un solo párrafo de ejemplo aplicado (nunca una definición). "
        "Debe mostrar cómo se usa o aplica el tema en una situación real o educativa."
    ))

    resp = llm.invoke([system, human])
    return resp.content.strip()

# 3) Validación de texto → solo LLM
@tool
def validacion_texto(texto_a_validar: str) -> str:
    """
    Valida gramática, coherencia y estilo; sugiere mejoras y devuelve versión corregida.
    Usa solo LLM (no Tavily).
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system = SystemMessage(content=(
        "Eres un corrector y editor. Revisa el texto en términos de ortografía, gramática, coherencia y estilo. "
        "Devuelve primero una breve nota (1-2 líneas) con observaciones, y luego una versión corregida del texto."
    ))
    resp = llm.invoke([system, HumanMessage(content=texto_a_validar)])
    return resp.content.strip()


# Lista de herramientas
tools = [comprension_texto, produccion_texto, validacion_texto]

# =========================================
# PROMPT BASE REACT
# =========================================
prompt = system_prompt = """
Eres EVA, una especialista en Comunicación.
Tu tarea es analizar la solicitud del usuario y decidir cuál herramienta usar.

- Si el usuario pide una definición, explicación o significado (por ejemplo: "qué es", "definición de", "concepto de"), usa la herramienta **comprension_texto**.
- Si el usuario pide un ejemplo, redacción o párrafo aplicado, usa la herramienta **produccion_texto**.
- Si el usuario pide que revises, corrijas o mejores un texto, usa la herramienta **validacion_texto**.

Responde siempre en formato JSON con los siguientes campos:
{
  "explicacion_profunda": "definición o explicación del tema",
  "parrafo_ejemplo": "ejemplo textual si aplica, o vacío si no aplica"
}
"""

# =========================================
# Crear el agente ReAct con herramientas
# =========================================
agent = create_react_agent(llm, tools, checkpointer=memory, prompt=prompt)

# =========================================
# FUNCIÓN PARA STREAMLIT
# =========================================
global_llm_with_tools = None  # Inicializar variable global

def get_comunicacion_agent():
    """Inicializa y devuelve el agente de Comunicación y su esquema."""
    global global_llm_with_tools

    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente Comunicación (LangGraph ReAct)...")
        global_llm_with_tools = agent
        print("✅ Agente Comunicación inicializado correctamente.")

    # Estructura esperada (el orquestador recibe 2 elementos)
    schema = {
        "explicacion_profunda": "str",
        "parrafo_ejemplo": "str"
    }

    return global_llm_with_tools, schema


