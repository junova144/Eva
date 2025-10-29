# 🌟 EVA: Soporte Educativo Autónomo 

**Plataforma Multi-Agente basada en LangGraph para la Asistencia educativa especializada y segura.**

---

## 💡 1. Resumen Ejecutivo y Propuesta de Valor (Negocios)

### El Desafío de la IA Educativa
Los modelos de lenguaje masivos (LLMs) fallan al proveer **precisión contextual** y **uniformidad pedagógica** en múltiples asignaturas. Depender de un único modelo para ser experto en Matemáticas, Comunicación e Inglés resulta en respuestas genéricas o superficiales.

### El Valor de EVA: Ruteo Especializado
EVA resuelve este desafío utilizando una arquitectura **Multi-Agente de Ruteo Estratégico** orquestada por **LangGraph**. El sistema actúa como un **Director Pedagógico** que:

1.  **Clasifica la Intención:** Identifica la materia y el tipo de solicitud del estudiante.
2.  **Rutea al Experto:** Envía la solicitud a un **Agente Especializado** (un Grafo de LangGraph) configurado con el *expertise* y las herramientas necesarias para esa asignatura.
3.  **Garantiza la Calidad:** El Agente utiliza capacidades avanzadas, como la **Búsqueda Híbrida**, para asegurar que la respuesta sea precisa y **pedagógicamente adecuada**.

> **EVA no es un *chatbot* unificado, es un equipo de expertos curriculares orquestados dinámicamente para maximizar la calidad de la respuesta educativa.**

---

## 🎨 2. Experiencia de Usuario y Componentes de Frontend

La interacción con EVA se realiza a través de una interfaz construida con Streamlit.

### 2.1. Interfaz Principal (`app/ui_streamlit.py`)
El **Frontend**, construido con **Streamlit**, ofrece una experiencia de *chat* conversacional. Su diseño enfatiza la **transparencia del ruteo**, mostrando mensajes clave sobre la decisión tomada por el Supervisor.

### 2.2. Contexto Curricular (`app/courses_data.py`)
Este módulo es la **Base de Conocimiento Estática** del sistema.
* **Función:** Almacena las taxonomías y los nombres **oficiales** de los cursos (ej. `"Ciencia y Tecnología"`).
* **Valor:** Asegura que el ruteo se alinee con la nomenclatura institucional.

### 2.3. Configuración (`app/config.py`)
Centraliza las variables de entorno, claves de API, y los parámetros de los LLMs. Es crucial para la **gestión de costos** y el **despliegue**.

---

## 📐 3. Arquitectura del Sistema: Orquestación y Agentes

El cerebro de EVA es un sistema de ejecución basado en **LangGraph**, divido en un Supervisor y múltiples Agentes Especializados.

### 3.1. El Supervisor Central (`main.py`)
El archivo principal actúa como el **Supervisor/Orquestador** y gestor de ruteo.
* **Ruteo Dinámico:** Contiene el diccionario **`AGENTS_EXECUTORS`** que mapea el `curso_detectado` a la instancia de **LangGraph Executor** correspondiente.
* **Función:** Llama primero al Validador y luego invoca el Grafo del Agente específico.

### 3.2. El Validador Estratégico (`app/validator.py`)
La primera línea de razonamiento del sistema.
* **Proceso:** Utiliza un LLM (e.g., GPT-4o) con un esquema **Pydantic** para convertir el lenguaje natural del usuario en un **comando de máquina estructurado (JSON)**.
* **Outputs Clave:** Genera el campo **`curso_detectado`** (para el ruteo) y la **`instruccion_maquina`** (el comando preciso para el agente).

### 3.3. Los Agentes Especializados (`agents/*.py`)
Cada archivo es un **Executor de LangGraph dedicado (Grafo)** que maneja una materia específica.
* **Estructura Interna:** Cada agente opera en un ciclo de **Razonamiento con Herramientas (ReAct-style)**.
* **Especialización:** Cada grafo tiene un *System Prompt* único que define su **tono pedagógico** y su enfoque experto.

---

## 🔧 4. Herramientas Integradas: Capacidad de los Expertos

Las herramientas (`tools/`) son las "manos" de los agentes, proporcionando capacidades avanzadas.

| Herramienta | Función Principal | Impacto en el Valor |
| :--- | :--- | :--- |
| **Producción Híbrida** | Implementa **RAG Híbrido**. Usa **Tavily Search** para contexto actualizado y luego un LLM anidado (GPT-4o-mini) para redacción. | **Actualidad y Eficiencia.** Garantiza respuestas precisas y optimiza los costos operativos. |
| **Comprensión/Validación** | Simula el análisis de la estructura y el nivel de complejidad del texto. | **Toma de Decisiones.** Permite al agente razonar sobre el nivel de profundidad requerido antes de generar la respuesta. |

### Stack Tecnológico
| Categoría | Componentes Clave |
| :--- | :--- |
| **Orquestación** | **LangGraph**, LangChain (Herramientas). |
| **Modelos/APIs** | **OpenAI API** (GPT-4o, GPT-4o-mini), **Tavily Search API**. |
| **Estructura** | **Pydantic** (Esquemas de Salida y Tipado), Python 3.10+. |
| **Frontend** | **Streamlit**. |

---

## ⚙️ 5. Instalación y Ejecución

Para levantar el proyecto EVA en tu entorno local.

### 5.1. Requisitos
* Python 3.10+
* Claves de API válidas para **OpenAI** y **Tavily Search**.

### 5.2. Setup de Entorno
```bash
# 1. Clonar el Repositorio
git clone [URL_DE_TU_REPOSITORIO]
cd EVA

# 2. Crear y Activar el Entorno Virtual
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Instalar Dependencias
pip install -r requirements.txt
```
¡Por supuesto! Lo corrijo. Es crucial que la guía de instalación y ejecución esté en formato de bloque de código para una copia rápida y sin errores.Aquí tienes la versión final con la Sección 5 (Instalación y Ejecución) completa dentro de un bloque de código Markdown:Markdown# 🌟 EVA: Educadora Virtual Autónoma

**Plataforma Multi-Agente basada en LangGraph para la Asistencia Curricular Precisa y Contextualizada.**

---

## 💡 1. Resumen Ejecutivo y Propuesta de Valor (Negocios)

### El Desafío de la IA Educativa
Los modelos de lenguaje masivos (LLMs) fallan al proveer **precisión contextual** y **uniformidad pedagógica** en múltiples asignaturas. Depender de un único modelo resulta en respuestas genéricas.

### El Valor de EVA: Ruteo Especializado
EVA utiliza una arquitectura **Multi-Agente de Ruteo Estratégico** orquestada por **LangGraph**. Actúa como un **Director Pedagógico** que clasifica la pregunta, la rutea a un **Agente Especializado** y garantiza la precisión mediante **Búsqueda Híbrida**.

> **EVA no es un *chatbot* unificado, es un equipo de expertos curriculares orquestados dinámicamente para maximizar la calidad de la respuesta educativa.**

---

## 🎨 2. Experiencia de Usuario y Componentes de Frontend

La interacción con EVA se realiza a través de una interfaz construida con Streamlit.

### 2.1. Interfaz Principal (`app/ui_streamlit.py`)
El **Frontend** de **Streamlit** ofrece una experiencia de *chat* conversacional, mostrando mensajes clave sobre la decisión del Supervisor.

### 2.2. Contexto Curricular (`app/courses_data.py`)
Módulo que almacena las taxonomías y los nombres **oficiales** de los cursos para asegurar la alineación del ruteo.

### 2.3. Configuración (`app/config.py`)
Centraliza las variables de entorno, claves de API, y los parámetros de los LLMs, crucial para la **gestión de costos** y el **despliegue**.

---

## 📐 3. Arquitectura del Sistema: Orquestación y Agentes

El cerebro de EVA se basa en **LangGraph** con un Supervisor y múltiples Agentes Especializados.

### 3.1. El Supervisor Central (`main.py`)
Actúa como el **Supervisor/Orquestador** y gestiona el ruteo. Contiene el diccionario **`AGENTS_EXECUTORS`** que mapea el `curso_detectado` a la instancia de **LangGraph Executor** correcta.

### 3.2. El Validador Estratégico (`app/validator.py`)
La primera línea de razonamiento. Utiliza un LLM con **Pydantic** para convertir el lenguaje natural en un **comando de máquina estructurado (JSON)** que define el **`curso_detectado`** y la **`instruccion_maquina`**.

### 3.3. Los Agentes Especializados (`agents/*.py`)
Cada archivo es un **Executor de LangGraph dedicado** que maneja una materia. Opera en un ciclo de **Razonamiento con Herramientas (ReAct-style)** y utiliza un *System Prompt* único para el **tono pedagógico** de su materia.

---

## 🔧 4. Herramientas Integradas: Capacidad de los Expertos

Las herramientas (`tools/`) extienden las capacidades de los agentes.

| Herramienta | Función Principal | Impacto en el Valor |
| :--- | :--- | :--- |
| **Producción Híbrida** | Implementa **RAG Híbrido** (Tavily Search + LLM) para contexto actualizado y redacción. | **Actualidad y Eficiencia.** Optimiza costos operativos. |
| **Comprensión/Validación** | Simula el análisis de la estructura y el nivel de complejidad del texto. | **Toma de Decisiones.** Permite al agente razonar sobre el nivel de profundidad requerido. |

### Stack Tecnológico
| Categoría | Componentes Clave |
| :--- | :--- |
| **Orquestación** | **LangGraph**, LangChain. |
| **Modelos/APIs** | **OpenAI API** (GPT-4o, GPT-4o-mini), **Tavily Search API**. |
| **Estructura** | **Pydantic**, Python 3.10+. |
| **Frontend** | **Streamlit**. |

---

## ⚙️ 5. Instalación y Ejecución

Para levantar el proyecto EVA en tu entorno local.

### 5.1. Requisitos
* Python 3.10+
* Claves de API válidas para **OpenAI** y **Tavily Search**.

### 5.2. Setup de Entorno
```bash
# 1. Clonar el Repositorio
git clone [URL_DE_TU_REPOSITORIO]
cd EVA

# 2. Crear y Activar el Entorno Virtual
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Instalar Dependencias
pip install -r requirements.txt
```
### 5.3. Configuración de Credenciales
Crea un archivo llamado **`.env`** en la raíz del proyecto para almacenar tus claves de API de forma segura:

```bash
# 1. Crear un archivo llamado .env en la raíz del proyecto
# 2. Añadir tus claves:
echo 'OPENAI_API_KEY="sk-..."' > .env
echo 'TAVILY_API_KEY="tvly-..."' >> .env
```

### 5.4. EjecuciónBash# Inicia la interfaz de usuario con Streamlit:
streamlit run app/ui_streamlit.py

# 📂 6. Estructura del RepositorioDirectorioFunción 
## 📂 6. Estructura del Repositorio

| Directorio/Archivo | Descripción |
| :--- | :--- |
| **EVA/** | Directorio Raíz del Proyecto. |
| ├── `main.py` | **Supervisor y Orquestador Principal.** Punto de entrada que llama al `validator` y rutea la ejecución al agente correcto (`AGENTS_EXECUTORS`). |
| ├── **app/** | Componentes de Interfaz, Flujo y Configuración. |
| │ ├── `ui_streamlit.py` | Interfaz principal de usuario (Frontend). |
| │ ├── `validator.py` | **Componente Clave:** Valida la entrada y genera el comando máquina (`curso_detectado`). |
| │ ├── `courses_data.py` | Base de conocimiento estática de cursos y grados. |
| │ └── `config.py` | Centraliza variables de entorno y parámetros de LLMs. |
| ├── **agents/** | Contiene los Grafos de LangGraph (Agentes Especializados). |
| │ ├── `agent_matematica.py` | Executor del Agente de Matemáticas. |
| │ ├── `agent_comunicacion.py` | Executor del Agente de Comunicación. |
| │ └── *[Otros Agentes]* | (ciencia, trabajo, inglés, etc.). |
| ├── **tools/** | Módulos con la definición de las **Herramientas personalizadas** (ej. RAG Híbrido). |
| ├── **data/** | Archivos auxiliares de contexto, *boosters*, o *datasets*. |
| ├── **logs/** | Archivos de registro y depuración del sistema. |
| └── `requirements.txt` | Lista de dependencias de Python necesarias. |
