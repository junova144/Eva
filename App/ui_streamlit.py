# ui_streamlit.py
import streamlit as st
import sys
import os
##pip install streamlit
# Agregamos la carpeta raíz (EVA) al sys.path para que Python encuentre main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import procesar_pregunta
from courses_data import cursos_por_grado, descripcion_cursos

# =========================
#   INTERFAZ PRINCIPAL
# =========================
def main():
    st.set_page_config(page_title="EVA - Asistente Educativo", page_icon="🤖", layout="centered")

    # Encabezado
    st.title("💡 EVA - Asistente Educativa Inteligente")
    st.markdown("Aprende, consulta y explora conocimientos según tu grado y curso escolar.")

    st.divider()

    # Selección de grado y curso
    col1, col2 = st.columns(2)
    with col1:
        grado = st.selectbox("📘 Selecciona tu grado:", list(cursos_por_grado.keys()))
    with col2:
        curso = st.selectbox("✏️ Selecciona tu curso:", cursos_por_grado[grado])

    st.markdown(f"**{curso} - {grado}**")
    st.info(descripcion_cursos.get(grado, {}).get(curso, "Descripción no disponible."))

    st.divider()

    # Sección de pregunta
    st.subheader("🤔 Escribe tu pregunta sobre el curso")
    pregunta = st.text_area("Tu pregunta:")

    if st.button("Enviar pregunta"):
        if pregunta.strip():
            with st.spinner("EVA está analizando tu pregunta..."):
                try:
                    respuesta = procesar_pregunta(pregunta, grado, curso)
                    st.markdown(respuesta)
                except Exception as e:
                    st.error(f"Ocurrió un error al procesar la pregunta: {e}")
        else:
            st.warning("Por favor, escribe una pregunta antes de enviar.")

    st.divider()
    st.caption("Desarrollado por Junova — Proyecto Final IA Generativa (EVA)")

# =========================
#   EJECUCIÓN PRINCIPAL
# =========================
if __name__ == "__main__":
    main()
