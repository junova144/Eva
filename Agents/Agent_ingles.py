# =======================================================================
# main.py - Agente Especialista en Inglés
# =======================================================================

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
# Importar BaseMessage, SystemMessage, ToolMessage desde .messages (es correcto)
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage 
# Importar MessagesPlaceholder desde .prompts (es la nueva ubicación)
from langchain_core.prompts import MessagesPlaceholder
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

# -----------------------------
# 0. Inicialización LLM y herramientas
# -----------------------------
llm_generador = ChatOpenAI(temperature=0.4, model="gpt-4o-mini")
tavily_tool = TavilySearchResults(max_results=3)
parser_generador = StrOutputParser()

# -----------------------------
# 1. Schema de salida
# -----------------------------
class RespuestaIngles(BaseModel):
    """Modelo Pydantic que define la estructura de la respuesta final del agente de Inglés."""
    explicacion_profunda: str = Field(description="Explicación detallada del tema de Inglés solicitado.")
    parrafo_ejemplo: str = Field(description="Ejemplo o ejercicio práctico que ilustra la explicación.")

parser_pydantic = PydanticOutputParser(pydantic_object=RespuestaIngles)
FORMAT_INSTRUCTIONS = parser_pydantic.get_format_instructions()

# -----------------------------
# 2. Herramientas disponibles
# -----------------------------
@tool
def generar_explicacion(input_text: str) -> str:
    """Genera explicación y ejemplo del tema de Inglés usando LLM."""
    prompt = f"Explica y da un ejemplo del tema: {input_text}"
    try:
        result = llm_generador.invoke({"input": prompt})
        return str(result)
    except Exception as e:
        return f"Error LLM: {e}"

@tool
def buscar_vocabulario(input_text: str) -> str:
    """Busca significado o contexto de palabras en inglés usando Tavily."""
    try:
        resultados = tavily_tool.invoke({"query": input_text})
        return "\n".join([r["content"] for r in resultados])
    except Exception as e:
        return f"Error Tavily: {e}"

@tool
def generar_practica(input_text: str) -> str:
    """Genera ejercicios cortos y sus soluciones sobre el tema de Inglés proporcionado."""
    prompt = f"Crea un ejercicio corto y su solución sobre: {input_text}"
    try:
        result = llm_generador.invoke({"input": prompt})
        return str(result)
    except Exception as e:
        return f"Error LLM: {e}"

tools = [generar_explicacion, buscar_vocabulario, generar_practica]

# -----------------------------
# 3. Grafo y estado
# -----------------------------
class InglesGraphState(TypedDict):
    """Estado del grafo del agente de Inglés: contiene mensajes acumulados."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

global_llm_with_tools = None

def ingles_agent_node(state: InglesGraphState):
    """Nodo principal del agente de Inglés que decide la respuesta final JSON."""
    messages = state["messages"]
    if global_llm_with_tools is None:
        raise ValueError("El agente de Inglés no ha sido inicializado.")

    final_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "ERES EVA, un Agente Especialista en Inglés para secundaria. "
            "Tu objetivo es responder la pregunta del estudiante de manera educativa y clara. "
            "Devuelve SOLO un JSON que cumpla con el formato Pydantic de salida."
        )),
        MessagesPlaceholder(variable_name="messages"),
        SystemMessage(content=FORMAT_INSTRUCTIONS)
    ])

    agent_chain = final_prompt | global_llm_with_tools
    response = agent_chain.invoke({"messages": messages})
    return {"messages": [response]}

def ingles_tool_node(state: InglesGraphState):
    """Ejecuta la herramienta solicitada por el agente y devuelve ToolMessage con el resultado."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_results: List[ToolMessage] = []

    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "generar_explicacion":
            result_content = generar_explicacion.invoke(tool_args)
        elif tool_name == "buscar_vocabulario":
            result_content = buscar_vocabulario.invoke(tool_args)
        elif tool_name == "generar_practica":
            result_content = generar_practica.invoke(tool_args)
        else:
            result_content = f"Error: Herramienta desconocida: {tool_name}"

        tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=result_content, name=tool_name))

    return {"messages": tool_results}

def ingles_should_continue(state: InglesGraphState) -> str:
    """Decide si continuar usando herramientas o terminar el flujo del agente."""
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END

def get_ingles_agent():
    """Inicializa y compila el agente especialista en Inglés, devuelve executor y schema Pydantic."""
    global global_llm_with_tools

    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente Especialista en Inglés...")
        llm_agent = ChatOpenAI(temperature=0, model="gpt-4o")
        global_llm_with_tools = llm_agent.bind_tools(tools)
        print("✅ Agente Inglés inicializado.")

    memory_saver = MemorySaver()
    workflow = StateGraph(InglesGraphState)
    workflow.add_node("agent", ingles_agent_node)
    workflow.add_node("tools", ingles_tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", ingles_should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=memory_saver), RespuestaIngles
