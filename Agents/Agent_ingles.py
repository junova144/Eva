# =======================================================================
# Agent_ingles.py - Agente Especialista en Inglés (EVA)
# =======================================================================

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

# 1) Explicación y ejemplo del tema
@tool
def generar_explicacion(tema: str) -> str:
    """
    Explica un tema de inglés (gramática, vocabulario o expresión)
    de forma clara y pedagógica, con un ejemplo breve al final.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    system = SystemMessage(content=(
        "Eres un profesor de inglés para secundaria. Explica el tema solicitado "
        "de forma sencilla y añade un ejemplo breve al final. No uses formato JSON."
    ))
    human = HumanMessage(content=f"Tema: {tema}")
    resp = llm.invoke([system, human])
    return resp.content.strip()


# 2) Búsqueda de vocabulario o significado contextual
@tool
def buscar_vocabulario(palabra: str) -> str:
    """
    Busca el significado y ejemplos de uso de una palabra o frase en inglés.
    Combina resultados web (Tavily) con una explicación educativa breve.
    """
    contexto = ""
    try:
        tavily = TavilySearchResults(max_results=3)
        raw_results = tavily.invoke({"query": f"meaning and examples of '{palabra}' in English"})
        if isinstance(raw_results, list):
            contexto = "\n".join([r.get("content", "") for r in raw_results if isinstance(r, dict)])
    except Exception as e:
        contexto = f"(No se pudo obtener contexto: {e})"

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.35)
    system = SystemMessage(content=(
        "Eres un profesor de inglés que explica vocabulario de forma contextual y sencilla. "
        "Resume los significados principales y da un ejemplo en inglés con su traducción al español."
    ))
    human = HumanMessage(content=f"Palabra o frase: {palabra}\n\nContexto web:\n{contexto}")
    resp = llm.invoke([system, human])
    return resp.content.strip()


# 3) Generación de ejercicios prácticos
@tool
def generar_practica(tema: str) -> str:
    """
    Crea un ejercicio corto (1–3 oraciones) con su solución
    sobre el tema o estructura gramatical indicada.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.45)
    system = SystemMessage(content=(
        "Eres un docente de inglés. Crea un ejercicio corto de práctica "
        "y proporciona la respuesta correcta. No des explicaciones teóricas."
    ))
    human = HumanMessage(content=f"Tema o estructura: {tema}")
    resp = llm.invoke([system, human])
    return resp.content.strip()


# Lista de herramientas
tools = [generar_explicacion, buscar_vocabulario, generar_practica]

# =========================================
# PROMPT BASE REACT
# =========================================
prompt = """
Eres EVA, una especialista en Inglés para secundaria.
Tu tarea es analizar la solicitud del estudiante y decidir qué herramienta usar.

- Si el usuario pide una explicación o definición de un tema, usa **generar_explicacion**.
- Si el usuario pide significado, traducción o uso de una palabra o frase, usa **buscar_vocabulario**.
- Si el usuario pide ejercicios o prácticas, usa **generar_practica**.

Responde siempre en formato JSON con los siguientes campos:
{
  "explicacion_profunda": "explicación o desarrollo del tema solicitado",
  "parrafo_ejemplo": "ejemplo, vocabulario o práctica generada"
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

def get_ingles_agent():
    """Inicializa y devuelve el agente de Inglés y su esquema."""
    global global_llm_with_tools

    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente de Inglés (LangGraph ReAct)...")
        global_llm_with_tools = agent
        print("✅ Agente Inglés inicializado correctamente.")

    schema = {
        "explicacion_profunda": "str",
        "parrafo_ejemplo": "str"
    }

    return global_llm_with_tools, schema

