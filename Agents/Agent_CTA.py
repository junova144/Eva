# Agents/Agent_CTA.py
import os
from typing import TypedDict, Annotated, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.tools import tool

# =======================================================================
# 0. INICIALIZACIÓN LLM
# =======================================================================
llm_generador = ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
parser_generador = StrOutputParser()

# =======================================================================
# 1. ESQUEMA DE SALIDA (PYDANTIC)
# =======================================================================
class RespuestaArgumentativa(BaseModel):
    explicacion_profunda: str = Field(description="Explicación detallada del concepto o fenómeno.")
    parrafo_ejemplo: str = Field(description="Ejemplo práctico o actividad que ilustra la explicación.")

parser_pydantic = PydanticOutputParser(pydantic_object=RespuestaArgumentativa)
FORMAT_INSTRUCTIONS = parser_pydantic.get_format_instructions()

# =======================================================================
# 2. HERRAMIENTAS (TOOLS) - CTA
# =======================================================================
class ExplicacionCientificaInput(BaseModel):
    concepto: str = Field(description="Fenómeno, proceso o concepto a explicar.")

@tool(args_schema=ExplicacionCientificaInput)
def explicacion_cientifica(concepto: str) -> str:
    """Explica un fenómeno natural, proceso biológico o físico de forma clara y correcta."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un profesor de Ciencias, Tecnología y Ambiente (CTA) para secundaria."),
        ("human", f"Explica de forma clara y correcta el siguiente concepto o fenómeno: {concepto}")
    ])
    cadena = prompt | llm_generador | parser_generador
    try:
        return cadena.invoke({})
    except Exception as e:
        return f"Error en explicacion_cientifica: {e}"


class ExperimentoSugeridoInput(BaseModel):
    concepto: str = Field(description="Fenómeno o concepto para el que se propone un experimento.")

@tool(args_schema=ExperimentoSugeridoInput)
def experimento_sugerido(concepto: str) -> str:
    """Propone un experimento o simulación sencilla para comprobar un concepto o fenómeno."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un profesor de CTA que sugiere experimentos educativos simples."),
        ("human", f"Propón un experimento o simulación sencilla para comprobar: {concepto}")
    ])
    cadena = prompt | llm_generador | parser_generador
    try:
        return cadena.invoke({})
    except Exception as e:
        return f"Error en experimento_sugerido: {e}"


class AnalisisImpactoInput(BaseModel):
    tema: str = Field(description="Tema ambiental o tecnológico a analizar.")

@tool(args_schema=AnalisisImpactoInput)
def analisis_impacto(tema: str) -> str:
    """Analiza impactos ambientales o tecnológicos y propone soluciones sostenibles."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un experto en sostenibilidad y CTA."),
        ("human", f"Analiza los impactos ambientales o tecnológicos de: {tema} y propone soluciones sostenibles.")
    ])
    cadena = prompt | llm_generador | parser_generador
    try:
        return cadena.invoke({})
    except Exception as e:
        return f"Error en analisis_impacto: {e}"


tools = [explicacion_cientifica, experimento_sugerido, analisis_impacto]

# =======================================================================
# 3. GRAFO Y ESTADO (LANGGRAPH) - AGENTE CTA
# =======================================================================
class CTAGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

global_llm_with_tools = None

def cta_agent_node(state: CTAGraphState):
    """Nodo principal del agente CTA que genera la respuesta final en formato JSON."""
    messages = state["messages"]
    if global_llm_with_tools is None:
        raise ValueError("El agente no ha sido inicializado. Ejecuta get_cta_agent() primero.")
    
    final_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "Eres un agente especialista en el curso de Ciencias, Tecnología y Ambiente para secundaria. "
            "Responde en formato JSON según el esquema provisto."
        )),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content=FORMAT_INSTRUCTIONS)
    ])
    
    agent_chain = final_prompt | global_llm_with_tools
    response = agent_chain.invoke({"messages": messages})
    return {"messages": [response]}


def cta_tool_node(state: CTAGraphState):
    """Ejecuta la herramienta llamada por el agente y devuelve ToolMessage con el resultado."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_results: List[ToolMessage] = []

    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "explicacion_cientifica":
            result_content = explicacion_cientifica.invoke(tool_args)
        elif tool_name == "experimento_sugerido":
            result_content = experimento_sugerido.invoke(tool_args)
        elif tool_name == "analisis_impacto":
            result_content = analisis_impacto.invoke(tool_args)
        else:
            result_content = f"Error: Herramienta desconocida: {tool_name}"

        tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=result_content, name=tool_name))

    return {"messages": tool_results}


def cta_should_continue(state: CTAGraphState) -> str:
    """Decide si el grafo debe continuar usando herramientas o terminar."""
    last_message = state["messages"][-1]
    return "tools" if getattr(last_message, "tool_calls", None) else END


def get_cta_agent():
    """Inicializa y compila el agente CTA; devuelve el executor y el schema Pydantic."""
    global global_llm_with_tools

    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente CTA...")
        llm_agent = ChatOpenAI(temperature=0, model="gpt-4o")
        global_llm_with_tools = llm_agent.bind_tools(tools)
        print("✅ Agente CTA inicializado.")

    memory_saver = MemorySaver()
    workflow = StateGraph(CTAGraphState)
    workflow.add_node("agent", cta_agent_node)
    workflow.add_node("tools", cta_tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", cta_should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=memory_saver), RespuestaArgumentativa
