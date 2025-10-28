# Agents/Agent_ept.py
import os
from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool

# =======================================================================
# 0. INICIALIZACIÓN DE LLMs Y HERRAMIENTAS DE SOPORTE
# =======================================================================

tavily_tool = TavilySearchResults(max_results=3)
llm_generador = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
parser_generador = StrOutputParser()

# =======================================================================
# 1. ESQUEMA DE SALIDA (PYDANTIC)
# =======================================================================

class RespuestaArgumentativa(BaseModel):
    explicacion_profunda: str = Field(description="Explicación detallada del concepto o proyecto.")
    parrafo_ejemplo: str = Field(description="Ejemplo práctico o estructura de proyecto que ilustra la explicación.")

parser_pydantic = PydanticOutputParser(pydantic_object=RespuestaArgumentativa)
FORMAT_INSTRUCTIONS = parser_pydantic.get_format_instructions()

# =======================================================================
# 2. HERRAMIENTAS (TOOLS) - EPT
# =======================================================================

class PlanProyectoInput(BaseModel):
    """Entrada para la herramienta que genera la estructura de un proyecto."""
    tema: str = Field(description="Tema o idea de proyecto a estructurar.")

@tool(args_schema=PlanProyectoInput)
def plan_proyecto(tema: str) -> str:
    """Genera la estructura completa de un proyecto educativo sobre un tema dado."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente pedagógico creativo para EPT, especializado en estructurar proyectos educativos."),
        ("human", f"Genera la estructura completa de un proyecto sobre: {tema}")
    ])
    cadena = prompt | llm_generador | parser_generador
    try:
        return cadena.invoke({})
    except Exception as e:
        return f"Error en plan_proyecto: {e}"

class ConceptoTecnologicoInput(BaseModel):
    """Entrada para explicar un concepto o herramienta tecnológica."""
    concepto: str = Field(description="Concepto o herramienta tecnológica a explicar.")

@tool(args_schema=ConceptoTecnologicoInput)
def concepto_tecnologico(concepto: str) -> str:
    """Explica de manera clara un concepto tecnológico y proporciona un ejemplo práctico."""
    search_query = f"Definición y ejemplos educativos: {concepto} EPT"
    try:
        results = tavily_tool.invoke({"query": search_query})
        contexto = "\n".join([f"- {r['content']}" for r in results])
    except Exception as e:
        contexto = f"(No se pudo obtener contexto externo: {e})"

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Eres un profesor de EPT que explica conceptos tecnológicos de forma pedagógica.\n{contexto}"),
        ("human", f"Explica de manera clara y pedagógica el concepto: {concepto} y proporciona un ejemplo práctico")
    ])
    cadena = prompt | llm_generador | parser_generador
    try:
        return cadena.invoke({})
    except Exception as e:
        return f"Error en concepto_tecnologico: {e}"

class EvaluacionProyectoInput(BaseModel):
    """Entrada para evaluar la viabilidad de un proyecto."""
    descripcion: str = Field(description="Descripción del proyecto a evaluar.")

@tool(args_schema=EvaluacionProyectoInput)
def evaluacion_proyecto(descripcion: str) -> str:
    """Evalúa la viabilidad del proyecto y sugiere mejoras pedagógicas."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un evaluador pedagógico de proyectos EPT."),
        ("human", f"Analiza la viabilidad de este proyecto y sugiere mejoras: {descripcion}")
    ])
    cadena = prompt | llm_generador | parser_generador
    try:
        return cadena.invoke({})
    except Exception as e:
        return f"Error en evaluacion_proyecto: {e}"

tools = [plan_proyecto, concepto_tecnologico, evaluacion_proyecto]

# =======================================================================
# 3. GRAFO Y ESTADO (LANGGRAPH) - AGENTE EPT
# =======================================================================

class EPTGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

global_llm_with_tools = None

def ept_agent_node(state: EPTGraphState):
    """Nodo principal del agente EPT que genera la respuesta final en formato JSON."""
    messages = state["messages"]
    if global_llm_with_tools is None:
        raise ValueError("El agente no ha sido inicializado. Ejecuta get_ept_agent() primero.")

    final_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Eres un asistente de Educación para el Trabajo (EPT) para secundaria."),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content=FORMAT_INSTRUCTIONS)
    ])

    agent_chain = final_prompt | global_llm_with_tools
    response = agent_chain.invoke({"messages": messages})
    return {"messages": [response]}

def ept_tool_node(state: EPTGraphState):
    """Ejecuta la herramienta llamada por el agente y devuelve ToolMessage con el resultado."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_results: List[ToolMessage] = []

    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "plan_proyecto":
            result_content = plan_proyecto.invoke(tool_args)
        elif tool_name == "concepto_tecnologico":
            result_content = concepto_tecnologico.invoke(tool_args)
        elif tool_name == "evaluacion_proyecto":
            result_content = evaluacion_proyecto.invoke(tool_args)
        else:
            result_content = f"Error: Herramienta desconocida: {tool_name}"

        tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=result_content, name=tool_name))

    return {"messages": tool_results}

def ept_should_continue(state: EPTGraphState) -> str:
    """Decide si el grafo debe continuar usando herramientas o terminar."""
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END


def get_ept_agent():
    """Inicializa y compila el agente EPT; devuelve el executor y el schema Pydantic."""
    global global_llm_with_tools

    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente EPT...")
        llm_agent = ChatOpenAI(temperature=0, model="gpt-4o")
        global_llm_with_tools = llm_agent.bind_tools(tools)
        print("✅ Agente EPT inicializado.")

    memory_saver = MemorySaver()
    workflow = StateGraph(EPTGraphState)
    workflow.add_node("agent", ept_agent_node)
    workflow.add_node("tools", ept_tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", ept_should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=memory_saver), RespuestaArgumentativa