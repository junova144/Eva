# =======================================================================
# Agents/Agent_matematica.py - Agente Especialista en Matemáticas
# =======================================================================

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

# -----------------------------
# 0. Inicialización LLM y soporte
# -----------------------------
tavily_tool = TavilySearchResults(max_results=3)
llm_generador = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
parser_generador = StrOutputParser()

# -----------------------------
# 1. Schema de salida
# -----------------------------
class RespuestaMatematica(BaseModel):
    """Modelo Pydantic para la respuesta del Agente Especialista en Matemáticas."""
    explicacion_profunda: str = Field(description="Explicación paso a paso del concepto o problema.")
    parrafo_ejemplo: str = Field(description="Ejemplo práctico o problema resuelto que ilustra la explicación.")

parser_pydantic = PydanticOutputParser(pydantic_object=RespuestaMatematica)
FORMAT_INSTRUCTIONS = parser_pydantic.get_format_instructions()

# -----------------------------
# 2. Herramientas Matemáticas
# -----------------------------
class ResolverInput(BaseModel):
    problema: str = Field(description="Problema matemático a resolver paso a paso.")

@tool(args_schema=ResolverInput)
def resolucion_problemas(problema: str) -> str:
    """Resuelve problemas matemáticos paso a paso y muestra la solución final."""
    system_prompt = (
        "Eres un asistente de matemáticas especializado en secundaria. "
        "Resuelve el problema paso a paso, mostrando cálculos y concluyendo con la respuesta final. "
        "Indica cómo verificar la solución si es posible."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Resuelve de forma clara y pedagógica este problema:\n\n{problema}")
    ])
    cadena = prompt | llm_generador | parser_generador
    try:
        return cadena.invoke({})
    except Exception as e:
        return f"Error en resolucion_problemas: {e}"

class ExplicacionInput(BaseModel):
    concepto: str = Field(description="Concepto matemático a explicar.")

@tool(args_schema=ExplicacionInput)
def explicacion_concepto(concepto: str) -> str:
    """Explica conceptos matemáticos con ejemplos usando LLM y contexto de Tavily."""
    search_query = f"Definición y ejemplos pedagógicos: {concepto} matemáticas secundaria"
    try:
        results = tavily_tool.invoke({"query": search_query})
        contexto = "\n".join([f"- {r['content']}" for r in results])
    except Exception as e:
        contexto = f"(No se pudo obtener contexto externo: {e})"

    system_prompt = (
        "Eres un profesor de matemáticas para secundaria. "
        "Usa el CONTEXTO cuando sea útil para ofrecer una explicación clara y concisa, "
        "incluye fórmulas si aplica y un ejemplo resuelto breve."
        f"\n\nCONTEXTO:\n{contexto}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Explica y da un ejemplo del concepto matemático: {concepto}")
    ])
    cadena = prompt | llm_generador | parser_generador
    try:
        return cadena.invoke({})
    except Exception as e:
        return f"Error en explicacion_concepto: {e}"

class VerificacionInput(BaseModel):
    enunciado: str = Field(description="Enunciado original o pasos del alumno.")
    respuesta_alumno: str = Field(description="Respuesta numérica o procedimiento del alumno.")

@tool(args_schema=VerificacionInput)
def verificacion_resultado(enunciado: str, respuesta_alumno: str) -> str:
    """Verifica la coherencia del resultado de un alumno y da retroalimentación pedagógica."""
    system_prompt = (
        "Eres un verificador pedagógico en matemáticas. Revisa el enunciado y la respuesta del alumno. "
        "Indica si es correcta, explica por qué o por qué no y sugiere pasos de corrección."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Enunciado: {enunciado}\nRespuesta del alumno: {respuesta_alumno}\nAnaliza y comenta.")
    ])
    cadena = prompt | llm_generador | parser_generador
    try:
        return cadena.invoke({})
    except Exception as e:
        return f"Error en verificacion_resultado: {e}"

tools = [resolucion_problemas, explicacion_concepto, verificacion_resultado]

# -----------------------------
# 3. Grafo y estado del Agente Matemáticas
# -----------------------------
class MatematicaGraphState(TypedDict):
    """Estado del grafo del Agente Especialista en Matemáticas."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

global_llm_with_tools = None

def matematica_agent_node(state: MatematicaGraphState):
    """Nodo principal del agente que genera la respuesta final JSON o decide usar herramientas."""
    messages = state["messages"]
    if global_llm_with_tools is None:
        raise ValueError("El agente de Matemáticas no ha sido inicializado.")

    final_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "ERES LUZIA-MATE, un Agente Especialista en Matemáticas para secundaria. "
            "Usa herramientas cuando necesites: 'resolucion_problemas', 'explicacion_concepto', "
            "y 'verificacion_resultado'. La respuesta final debe ser un JSON que cumpla el formato Pydantic."
        )),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content=FORMAT_INSTRUCTIONS)
    ])
    agent_chain = final_prompt | global_llm_with_tools
    response = agent_chain.invoke({"messages": messages})
    return {"messages": [response]}

def matematica_tool_node(state: MatematicaGraphState):
    """Ejecuta la herramienta solicitada y devuelve ToolMessage con el resultado."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_results: List[ToolMessage] = []

    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "resolucion_problemas":
            result_content = resolucion_problemas.invoke(tool_args)
        elif tool_name == "explicacion_concepto":
            result_content = explicacion_concepto.invoke(tool_args)
        elif tool_name == "verificacion_resultado":
            result_content = verificacion_resultado.invoke(tool_args)
        else:
            result_content = f"Error: Herramienta desconocida: {tool_name}"

        tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=result_content, name=tool_name))

    return {"messages": tool_results}

def matematica_should_continue(state: MatematicaGraphState) -> str:
    """Decide si el flujo del grafo continúa usando herramientas o termina."""
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END

def get_matematica_agent():
    """Inicializa y compila el Agente Especialista en Matemáticas; devuelve executor y schema Pydantic."""
    global global_llm_with_tools

    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente Matemáticas...")
        llm_agent = ChatOpenAI(temperature=0, model="gpt-4o")
        global_llm_with_tools = llm_agent.bind_tools(tools)
        print("✅ Agente Matemáticas inicializado.")

    memory_saver = MemorySaver()
    workflow = StateGraph(MatematicaGraphState)
    workflow.add_node("agent", matematica_agent_node)
    workflow.add_node("tools", matematica_tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", matematica_should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=memory_saver), RespuestaMatematica
