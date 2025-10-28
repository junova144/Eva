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

# Inicializar Tavily Search
tavily_tool = TavilySearchResults(max_results=3)

# Inicializar LLM generador dedicado
llm_generador = ChatOpenAI(temperature=0.4, model="gpt-4o-mini")
parser_generador = StrOutputParser()

# =======================================================================
# 1. ESQUEMA DE SALIDA (PYDANTIC)
# =======================================================================

class RespuestaArgumentativa(BaseModel):
    """Estructura de salida validada para la explicación del tema y ejemplo."""
    explicacion_profunda: str = Field(
        description="Explicación detallada de la naturaleza, estructura y elementos clave del tema."
    )
    parrafo_ejemplo: str = Field(
        description="Párrafo de ejemplo bien redactado y claro para ilustrar la tesis o el concepto."
    )

parser_pydantic = PydanticOutputParser(pydantic_object=RespuestaArgumentativa)
FORMAT_INSTRUCTIONS = parser_pydantic.get_format_instructions()

# =======================================================================
# 2. HERRAMIENTAS (TOOLS)
# =======================================================================

class ComprensionInput(BaseModel):
    """Herramienta para analizar y clasificar la estructura, tono y propósito del texto de entrada."""
    texto_a_analizar: str = Field(description="Texto completo proporcionado por el usuario o el agente para su análisis.")

@tool(args_schema=ComprensionInput)
def comprension_texto(texto_a_analizar: str) -> str:
    """Simula el análisis de estructura (narrativo, argumentativo, expositivo) y elementos clave."""
    if "tesis" in texto_a_analizar.lower() or "postura" in texto_a_analizar.lower():
        return "Análisis Estructural (Simulación): Texto de naturaleza **Argumentativa**. Se identifican Tesis y Argumentos."
    elif len(texto_a_analizar.split()) < 10:
        return "Análisis Estructural (Simulación): Texto muy breve. El alumno busca una definición concisa. Tono directo."
    else:
        return "Análisis Estructural (Simulación): Texto descriptivo general. La estructura es narrativa o expositiva. Tono informativo."


class ProduccionInput(BaseModel):
    """Herramienta para generar un fragmento de texto de ejemplo basado en un tema/tipo."""
    tema_o_tipo_texto: str = Field(description="El tema o tipo de texto que el agente necesita generar (ej. 'párrafo de conclusión', 'poema', 'ejemplo de carta').")

@tool(args_schema=ProduccionInput)
def produccion_texto(tema_o_tipo_texto: str) -> str:
    """Genera contenido real (RAG) usando Tavily para contexto y un LLM anidado para redacción."""
    search_query = f"Ejemplo educativo de Comunicación: {tema_o_tipo_texto}"
    try:
        contexto_tavily = tavily_tool.invoke({"query": search_query})
        contexto_formateado = "\n".join([f"- {r['content']}" for r in contexto_tavily])
    except Exception as e:
        contexto_formateado = f"Error al buscar en Tavily: {e}. Usando solo conocimiento interno del LLM."

    system_prompt = (
        "Eres un redactor educativo especializado. Genera un contenido conciso, claro y preciso para secundaria. "
        "Utiliza el CONTEXTO DE BÚSQUEDA para asegurar precisión. Responde ÚNICAMENTE con el texto solicitado. "
        f"\n\nCONTEXTO DE BÚSQUEDA:\n{contexto_formateado}"
    )

    prompt_generacion = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", f"Genera el siguiente fragmento de texto educativo: {tema_o_tipo_texto}")
    ])

    cadena_generacion = prompt_generacion | llm_generador | parser_generador
    try:
        contenido_real = cadena_generacion.invoke({})
        return f"Contenido Generado: {contenido_real.strip()}"
    except Exception as e:
        return f"Contenido Generado: Fallo del LLM interno de la herramienta. Error: {e}"


class ValidacionInput(BaseModel):
    """Herramienta para validar la coherencia, gramática y estilo de un texto."""
    texto_a_validar: str = Field(description="El texto que el agente o el alumno quieren verificar.")

@tool(args_schema=ValidacionInput)
def validacion_texto(texto_a_validar: str) -> str:
    """Valida la gramática, coherencia y estilo de un texto (Simulación)."""
    if len(texto_a_validar.split()) < 5:
        return "Validación (Simulación): Texto demasiado corto. Posiblemente requiere expansión."
    elif "coherencia" in texto_a_validar.lower() or "pero" in texto_a_validar.lower():
        return "Validación (Simulación): La coherencia es adecuada, la puntuación es correcta, pero revisa el uso de conectores para transiciones más fluidas."
    else:
        return "Validación (Simulación): El texto es gramatical y ortográficamente correcto. Estilo adecuado al nivel secundario."

tools = [comprension_texto, produccion_texto, validacion_texto]

# =======================================================================
# 3. GRAFO Y ESTADO (LANGGRAPH)
# =======================================================================

class CommunicationGraphState(TypedDict):
    """Representa el estado del grafo (memoria y contexto)."""
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

global_llm_with_tools = None

def communication_agent_node(state: CommunicationGraphState):
    """Nodo que decide si usar una herramienta o si genera la respuesta final JSON."""
    messages = state["messages"]
    if global_llm_with_tools is None:
        raise ValueError("El agente no ha sido inicializado. Ejecuta get_comunicacion_agent() primero.")

    final_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "ERES EVA, un Agente Especialista en Comunicación. Tu objetivo es responder la pregunta del estudiante. "
            "**Utiliza tu herramienta 'produccion_texto' siempre que debas generar explicaciones detalladas o ejemplos** (necesarios para el JSON de salida), ya que esa herramienta incorpora búsqueda web (Tavily) y generación de contenido real. "
            "**REGLA CRÍTICA:** Tu respuesta final DEBE ser un objeto JSON que se ajuste estrictamente al formato Pydantic. "
            "Una vez que tengas la respuesta, emite SOLO el JSON final."
        )),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content=FORMAT_INSTRUCTIONS)
    ])

    agent_chain = final_prompt | global_llm_with_tools
    response = agent_chain.invoke({"messages": messages})
    return {"messages": [response]}


def communication_tool_node(state: CommunicationGraphState):
    """Nodo que ejecuta la herramienta llamada por el Agente."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "comprension_texto":
            result_content = comprension_texto.invoke(tool_args)
        elif tool_name == "produccion_texto":
            result_content = produccion_texto.invoke(tool_args)
        elif tool_name == "validacion_texto":
            result_content = validacion_texto.invoke(tool_args)
        else:
            result_content = f"Error: Herramienta desconocida: {tool_name}"

        tool_results.append(ToolMessage(tool_call_id=tool_call["id"], content=result_content, name=tool_name))

    return {"messages": tool_results}


def communication_should_continue(state: CommunicationGraphState) -> str:
    """Router: Decide si continuar el ciclo (usar herramientas) o terminar (END)."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


def get_comunicacion_agent():
    """Compila y devuelve el ejecutor del agente de Comunicación y su Schema de respuesta."""
    global global_llm_with_tools

    if global_llm_with_tools is None:
        print("🤖 Inicializando Agente Comunicación...")
        llm_agent = ChatOpenAI(temperature=0, model="gpt-4o")
        global_llm_with_tools = llm_agent.bind_tools(tools)
        print("✅ Agente Comunicación inicializado.")

    memory_saver = MemorySaver()

    workflow_comunicacion = StateGraph(CommunicationGraphState)
    workflow_comunicacion.add_node("agent", communication_agent_node)
    workflow_comunicacion.add_node("tools", communication_tool_node)
    workflow_comunicacion.set_entry_point("agent")

    workflow_comunicacion.add_conditional_edges(
        "agent",
        communication_should_continue,
        {"tools": "tools", END: END}
    )

    workflow_comunicacion.add_edge("tools", "agent")

    return workflow_comunicacion.compile(checkpointer=memory_saver), RespuestaArgumentativa
