# app/validador.py
# =====================================================
# 🔹 EVA - Cadena Modular de Validación y Generación de Prompt
# =====================================================

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from typing import Dict, Any
import json
import os

# ----------------------------------------------------
# 1. INICIALIZACIÓN DE COMPONENTES (GLOBAL)
# ----------------------------------------------------
llm_validator = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    verbose=False  ##(antes True, genera logs innecesarios)
)

parser = StrOutputParser()

# =======================================================================
# 2. DEFINICIÓN DE CADENAS LCEL
# ======================================================================

########## Cadena (Detección de Curso)
curso_prompt = ChatPromptTemplate.from_template("""
Eres un analizador de preguntas escolares.

Solo considera estos cursos: Matemática, Comunicación, Ciencia y Tecnología, Educación para el Trabajo, Inglés.

Analiza la siguiente pregunta y devuelve únicamente el curso correspondiente de la lista anterior.
Devuelve solo uno de: Matemática, Comunicación, Ciencia y Tecnología, Educación para el Trabajo, Inglés.

Pregunta: {pregunta}
""")
curso_chain = curso_prompt | llm_validator | parser

# ----------------------------------------------------
# Función de normalización de texto de curso

#  agregado para mejorar coincidencias entre cursos con variantes menores
def normalizar_curso(texto: str) -> str:
    return texto.lower().replace(".", "").strip()

########### Cadena (Contraste / Python Pura)
def generar_contraste_binario_estructurado(datos: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compara solo la coherencia del curso. Asume que la intención siempre es GUIA.
    """
    curso_sistema = datos.get("curso_sistema", "")
    curso_detectado = datos.get("curso_detectado", "")

    curso_coincide = (normalizar_curso(curso_sistema) == normalizar_curso(curso_detectado))
    valido = curso_coincide  # La pregunta es válida si el curso es correcto.

    # --- Generación de Código de Error ---
    if valido:
        codigo_error = "OK"
        mensaje_base = "OK"
    else:
        codigo_error = "ERROR_CURSO_INCORRECTO"
        mensaje_base = "CURSO_INCORRECTO"

    return {
        "valido": valido,
        "codigo_error": codigo_error,
        "mensaje_base": mensaje_base,
        **datos  # Devolvemos todo el contexto
    }

contraste_chain = RunnableLambda(generar_contraste_binario_estructurado)

########## Cadena 4 (Generador de Prompt Final)
prompt_especializado = ChatPromptTemplate.from_template("""
ROL: Eres un generador de respuestas finales del sistema LuzIA. Tu ÚNICA FUNCIÓN es formatear el resultado basado en el diagnóstico.

INSTRUCCIONES CRÍTICAS:
1. Devuelve el resultado final en el formato JSON requerido.
2. Solo se admite el fallo por curso incorrecto.

DATOS DE ENTRADA:
- Validez: {valido}
- Tipo de Fallo: {mensaje_base}
- Curso del sistema: {curso_sistema}
- Curso detectado: {curso_detectado}
- Pregunta original: {entrada_usuario}

------------------------------
1️⃣ Si **valido es False**:
   Genera una respuesta breve y amable dirigida al usuario.
   
   - SI **Tipo de Fallo es CURSO_INCORRECTO** (el único fallo posible): 
        Devuelve el texto: 
        "La pregunta no corresponde al curso de **{curso_sistema}**. 
        Fue clasificada como **{curso_detectado}**. 
        Por favor, reformula tu pregunta dentro del contexto de **{curso_sistema}**."
------------------------------
2️⃣ Si **valido es True** (Si mensaje_base es 'OK'):
   Genera una instrucción de máquina limpia para el agente LLM especialista:
   [COMANDO_AGENTE]
   ANALIZA_TEMA: {entrada_usuario}
   CONTEXTO_EDUCATIVO: {curso_sistema}
   ACCIÓN: Generar respuesta pedagógica, clara y precisa.
------------------------------
""")  # 🔁 REEMPLAZADO (corrige los placeholders de {curso_sistema}/{curso_detectado})

generar_prompt_agente = prompt_especializado | llm_validator | parser

# =======================================================================
# 3. COMPILACIÓN DEL PIPELINE GLOBAL
# =======================================================================

#  RunnableLambda (una sola tarea)
deteccion_chain = RunnableLambda(
    lambda x: {"curso_detectado": curso_chain.invoke({"pregunta": x["entrada_usuario"]}).strip()}
)

pipeline_decision = RunnableSequence(
    RunnableLambda(lambda x: {**x, **deteccion_chain.invoke(x)}),
    contraste_chain
)

# =======================================================================
# 4. FUNCIÓN WRAPPER FINAL
# =======================================================================
def run_eva_pipeline(grado_sistema: str, curso_sistema: str, pregunta: str) -> Dict:
    """
    Ejecuta el pipeline LCEL simplificado (C2 -> C3) y luego la Cadena 4.
    """
    input_pipeline = {
        "entrada_usuario": pregunta,
        "grado_sistema": grado_sistema,
        "curso_sistema": curso_sistema,
    }

    # 1. Ejecutar la Detección y Contraste (C2 -> C3)
    resultado_decision = pipeline_decision.invoke(input_pipeline)

    # 2. Ejecutar la Generación Final (Cadena 4)
    texto_final = generar_prompt_agente.invoke(resultado_decision).strip()

    # 3. Formatear la Salida para el sistema
    es_valido = resultado_decision.get("valido", False)

    if es_valido:
        validacion = {"valido": True, "mensaje": "Validación exitosa."}
        prompt_final = texto_final
    else:
        validacion = {"valido": False, "mensaje": texto_final}
        prompt_final = ""

    # 🔁 REEMPLAZADO: devuelve el JSON como objeto, no string, más manejable desde Streamlit
    return {
        "prompt_final": prompt_final,
        "validacion": validacion,
        "curso_final": curso_sistema
    }

# =====================================================
# FIN DEL MÓDULO
# =====================================================
