import time
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sys

# Intentar importar Plotly; si no está disponible, mostrar mensaje amigable
PLOTLY_AVAILABLE = True
_plotly_import_err = None
try:
    import plotly.graph_objects as go
    import plotly.express as px  # opcional, por si lo necesitas después
except Exception as _e:
    PLOTLY_AVAILABLE = False
    _plotly_import_err = str(_e)


# --------------------------------------------------------------------
# Helper: scroll suave a un elemento con id dado (parallax-like)
# --------------------------------------------------------------------
def scroll_to_here(delay_ms: int = 200, key: str | None = None) -> None:
    """Hace scroll suave a un elemento con id == key después de delay_ms ms."""
    if key is None:
        return

    js = f"""
    <script>
    function _st_scroll() {{
        const el = window.parent.document.getElementById("{key}");
        if (el) {{
            el.scrollIntoView({{behavior: 'smooth', block: 'start'}});
        }} else {{
            window.parent.scrollTo({{top: 0, behavior: 'smooth'}});
        }}
    }}
    setTimeout(_st_scroll, {delay_ms});
    </script>
    """
    components.html(js, height=0)


# ---------------------------------------------------------
# Configuración básica de la página
# ---------------------------------------------------------
# theme will be set via .streamlit/config.toml instead
st.set_page_config(
    page_title="Place Game - Evaluación de Lugar Dummy",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Estado de la app
# ---------------------------------------------------------
if "paso" not in st.session_state:
    st.session_state.paso = 0  # índice del atributo actual (0..3)

if "scroll_target" not in st.session_state:  # int (índice de sección) o "resultados"
    st.session_state.scroll_target = None

if "mostrar_grafica" not in st.session_state:  # pantalla "Estamos preparando..."
    st.session_state.mostrar_grafica = False

# Nuevos campos: nombre del lugar, evaluador, timestamp
if "nombre_lugar" not in st.session_state:
    st.session_state.nombre_lugar = ""

if "nombre_evaluador" not in st.session_state:
    st.session_state.nombre_evaluador = ""

if "timestamp_eval" not in st.session_state:
    st.session_state.timestamp_eval = None

# Bytes de la gráfica PNG
if "grafica_png_bytes" not in st.session_state:
    st.session_state.grafica_png_bytes = None

# ---------------------------------------------------------
# Definición de atributos, máximos y preguntas (Place Game)
# ---------------------------------------------------------
atributos = {
    "Encuentro": {
        "max": 32,
        "preguntas": [
            "¿Funciona como punto de encuentro?",
            "¿Hay evidencia de vecinos organizados?",
            "¿Existe sentimiento de orgullo y propiedad?",
            "¿Hay niños y niñas presentes?",
            "¿Hay personas mayores presentes?",
            "¿Hay familias presentes?",
            "¿Hay mujeres presentes?",
            "¿Las personas presentes están utilizando el espacio de diferentes maneras?",
        ],
    },
    "Usos y Actividades": {
        "max": 28,
        "preguntas": [
            "¿Hay opciones de actividades por hacer?",
            "¿Hay personas y permanecen tiempo en él?",
            "¿Los niños y niñas se ven divertidos?",
            "¿El mobiliario es útil y funcional?",
            "¿El espacio es amigable con el medio ambiente?",
            "¿Hay una mezcla de comercios y servicios?",
            "¿Hay vitalidad económica del área?",
        ],
    },
    "Conexiones y Accesos": {
        "max": 24,
        "preguntas": [
            "¿Está conectado con su alrededor?",
            "¿Está en una ubicación conveniente?",
            "¿Existen paradas de transporte público?",
            "¿Se puede caminar cómodamente?",
            "¿Hay rampas para sillas de ruedas?",
            "¿Hay señalización adecuada?",
        ],
    },
    "Comodidad e Imagen": {
        "max": 24,
        "preguntas": [
            "¿Es un lugar atractivo?",
            "¿Es un lugar agradable?",
            "¿Parece un sitio seguro?",
            "¿Está limpio y mantenido?",
            "¿Hay sitios cómodos para sentarse?",
            "¿Es un lugar cómodo en general?",
        ],
    },
}

# ---------------------------------------------------------
# Paleta de colores para los atributos (gráfica final)
# ---------------------------------------------------------
palette = {
    "Encuentro": "#ac80ab",        # morado
    "Usos y Actividades": "#f7ad79",  # naranja
    "Conexiones y Accesos": "#00b2d1",  # azul
    "Comodidad e Imagen": "#6d967b",   # verde
}
color_fondo = "#ecf0f1"  # gris claro para la parte no alcanzada
orden_atributos = list(atributos.keys())

# ---------------------------------------------------------
# Estilos: bloques de atributos + fuentes y widgets
# ---------------------------------------------------------
css = """
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght;700&display=swap" rel="stylesheet">
<style>
:root{
    --primary-color: #2ecc71;
    --accent-1: #4b295e;   /* Encuentro */
    --accent-2: #ec6420;   /* Usos y Actividades */
    --accent-3: #0087c4;   /* Conexiones y Accesos */
    --accent-4: #1c301d;   /* Comodidad e Imagen */
    --slider-track: #e6eef0;
    --slider-thumb: #2c3e50;
    --progress-bg: #ecf0f1;
    --progress-fill: #2ecc71;
}

/* ======== FORZAR TEMA CLARO (LIGHT) ======== */
/* Contenedor principal de la app */
[data-testid="stAppViewContainer"] {
    background-color: #FFFFFF !important;
    color: #262730 !important;  /* texto oscuro */
}

/* Sidebar en claro */
[data-testid="stSidebar"] {
    background-color: #F0F2F6 !important;
    color: #262730 !important;
}

/* Fondo de los bloques principales */
section.main > div {
    background-color: #FFFFFF !important;
}

/* Botones en esquema claro */
.stButton > button {
    background-color: #2ecc71 !important;
    color: #FFFFFF !important;
    border-radius: 4px !important;
    border: none !important;
}
.stButton > button:hover {
    background-color: #27ae60 !important;
}

/* ======== FIN FORZAR TEMA CLARO ======== */

/* Fuente global Poppins */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif!important;
}

/* Bloques de atributo */
.bloque-atributo {
    border-left: 6px solid #ccc;
    padding-left: 10px;
    margin-bottom: 1.5rem;
}
.bloque-encuentro   { border-color: var(--accent-1); }
.bloque-usos        { border-color: var(--accent-2); }
.bloque-conexiones  { border-color: var(--accent-3); }
.bloque-comodidad   { border-color: var(--accent-4); }

.bloque-atributo h3 {
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.bloque-encuentro h3   { color: var(--accent-1); }
.bloque-usos h3        { color: var(--accent-2); }
.bloque-conexiones h3  { color: var(--accent-3); }
.bloque-comodidad h3   { color: var(--accent-4); }

/* Sliders */
input[type="range"] {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 10px;
    background: var(--slider-track);
    border-radius: 8px;
    outline: none;
    margin: 12px 0;
}
input[type="range"]::-webkit-slider-runnable-track {
    height: 10px;
    border-radius: 8px;
    background: var(--slider-track);
}
input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: var(--slider-thumb);
    margin-top: -5px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
input[type="range"]:focus {
    outline: none;
}
/* Firefox */
input[type="range"]::-moz-range-track {
    height:10px;
    background:var(--slider-track);
    border-radius:8px;
}
input[type="range"]::-moz-range-thumb {
    width:20px;
    height:20px;
    background:var(--slider-thumb);
    border-radius:50%;
    box-shadow:0 2px 6px rgba(0,0,0,0.15);
}

/* Progress bar */
div[data-testid="stProgress"] div[role="progressbar"] {
    background: var(--progress-bg)!important;
    border-radius: 8px;
    height: 18px;
}
div[data-testid="stProgress"] div[role="progressbar"] > div {
    background: var(--progress-fill)!important;
    border-radius: 8px;
    height: 18px;
}

/* Labels */
.stMetric > div.stMarkdown.stText,
label {
    font-family: 'Poppins', sans-serif!important;
}
label[for] {
    font-weight: 600;
    font-size: 14px;
}

/* Estilos de impresión: ocultar menú y pie de Streamlit */
@media print {
    header[data-testid="stHeader"],
    footer,
    div[data-testid="stToolbar"] {
        display: none!important;
    }
}
</style>
"""
components.html(css, height=0)

components.html(css, height=0)

# ---------------------------------------------------------
# Título principal
# ---------------------------------------------------------
st.title("Place Game - Evaluación de Lugar Dummy")
st.write(
    "Ajusta los sliders para calificar cada pregunta de 0 a 4 puntos. Evalúa la Excelencia de cualquier lugar de acuerdo a la rueda del Place Game. Hecha por José Bucio para demostración."
    
)

# ---------------------------------------------------------
# Datos generales de la evaluación
# ---------------------------------------------------------
st.subheader("Datos de la evaluación")
col_lugar, col_eval = st.columns(2)

with col_lugar:
    st.session_state.nombre_lugar = st.text_input(
        "Nombre del lugar",
        value=st.session_state.nombre_lugar,
        placeholder="Ej. Parque México, Reggio Emilia, etc.",
    )

with col_eval:
    st.session_state.nombre_evaluador = st.text_input(
        "Nombre de quien evalúa",
        value=st.session_state.nombre_evaluador,
        placeholder="Ej. Fred Kent, Jane Jacobs, etc.",
    )

st.markdown("---")

# ---------------------------------------------------------
# Barra de progreso (parte superior)
# ---------------------------------------------------------
num_secciones = len(orden_atributos)
secciones_completadas = min(st.session_state.paso, num_secciones)
progreso = int(secciones_completadas / num_secciones * 100) if num_secciones > 0 else 0

st.progress(
    progreso,
    text=f"Progreso: {secciones_completadas} de {num_secciones} secciones completadas",
)

# ---------------------------------------------------------
# Cuestionario paso a paso
# ---------------------------------------------------------
st.header("Cuestionario")

if num_secciones > 0:
    visible_hasta = min(st.session_state.paso, num_secciones - 1)
else:
    visible_hasta = -1

for idx in range(visible_hasta + 1):
    nombre_atributo = orden_atributos[idx]
    info = atributos[nombre_atributo]

    # Clase CSS según el atributo
    if nombre_atributo == "Encuentro":
        clase = "bloque-atributo bloque-encuentro"
    elif nombre_atributo == "Usos y Actividades":
        clase = "bloque-atributo bloque-usos"
    elif nombre_atributo == "Conexiones y Accesos":
        clase = "bloque-atributo bloque-conexiones"
    else:  # "Comodidad e Imagen"
        clase = "bloque-atributo bloque-comodidad"

    # Contenedor con id para scroll
    st.markdown(f'<div id="sec_{idx}" class="{clase}">', unsafe_allow_html=True)
    st.markdown(
        f"### {nombre_atributo} "
    )

    cols = st.columns(2)
    subtotal = 0

    # Sliders de este atributo
    for i, pregunta in enumerate(info["preguntas"]):
        col = cols[i % 2]
        key = f"{nombre_atributo}_{i}"
        with col:
            valor_inicial = st.session_state.get(key, 0)
            valor = st.slider(
                label=pregunta,
                min_value=0,
                max_value=4,
                value=valor_inicial,
                step=1,
                key=key,
            )
            subtotal += valor



# ---------------------------------------------------------
# Botones de navegación (con scroll automático)
# ---------------------------------------------------------
if st.session_state.paso < num_secciones:
    # Aún estamos en el cuestionario
    if st.session_state.paso < num_secciones - 1:
        # No es la última sección: avanzamos a la siguiente
        if st.button("Siguiente sección", key="siguiente_seccion"):
            st.session_state.paso += 1
            st.session_state.scroll_target = st.session_state.paso  # int
            st.rerun()
    else:
        # Última sección visible; siguiente paso: resultados
        if st.button("Ver resultados", key="ver_resultados"):
            st.session_state.paso += 1  # paso == num_secciones
            st.session_state.scroll_target = "resultados"
            st.session_state.mostrar_grafica = False  # forzar pantalla de carga
            st.rerun()

# ---------------------------------------------------------
# Resultados (pantalla de carga + gráfica + interpretación)
# ---------------------------------------------------------
if st.session_state.paso >= num_secciones and num_secciones > 0:
    # Marca de tiempo (se fija al llegar a resultados por primera vez)
    if st.session_state.timestamp_eval is None:
        st.session_state.timestamp_eval = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    timestamp_eval = st.session_state.timestamp_eval
    nombre_lugar = st.session_state.nombre_lugar or "No especificado"
    nombre_evaluador = st.session_state.nombre_evaluador or "No especificado"

    # Reconstruir subtotales y respuestas desde session_state
    subtotales = {}
    respuestas = {}

    for nombre_atributo, info in atributos.items():
        subtotal = 0
        respuestas[nombre_atributo] = []
        for i, _ in enumerate(info["preguntas"]):
            key = f"{nombre_atributo}_{i}"
            valor = st.session_state.get(key, 0)
            subtotal += valor
            respuestas[nombre_atributo].append(valor)
        subtotales[nombre_atributo] = subtotal

    # Totales globales
    total_puntos = sum(subtotales.values())
    max_total = sum(a["max"] for a in atributos.values())

    # Radios del anillo de datos
    radio_interno = 6
    radio_externo = 25

    # Fracciones normalizadas 0–1 por atributo
    valores_norm = {}
    for nombre_atributo, info in atributos.items():
        max_attr = info["max"]
        if max_attr > 0:
            fraccion = subtotales[nombre_atributo] / max_attr
            valores_norm[nombre_atributo] = fraccion
        else:
            valores_norm[nombre_atributo] = 0.0

    # Ancla para resultados
    st.markdown('<div id="resultados"></div>', unsafe_allow_html=True)

    # Scroll a resultados si fue solicitado
    if st.session_state.scroll_target == "resultados":
        scroll_to_here(200, key="resultados")
        st.session_state.scroll_target = None

    st.header("Visualización del lugar")
    st.write(
        f"**Lugar:** {nombre_lugar} | "
        f"**Evaluado por:** {nombre_evaluador} | "
        f"**Fecha y hora de evaluación:** {timestamp_eval}"
    )

    # Pantalla intermedia: "Estamos preparando tus resultados"
    if not st.session_state.mostrar_grafica:
        placeholder = st.empty()
        with placeholder.container():
            st.subheader("Estamos preparando tus resultados")
            st.write(
                "Por favor espera unos segundos mientras generamos la visualización del lugar."
            )
        with st.spinner("Calculando..."):
            time.sleep(1.2)  # tiempo artificial
        st.session_state.mostrar_grafica = True
        st.rerun()

    # -----------------------------------------------------
    # Gráfica final tipo "rueda"
    # -----------------------------------------------------
    if PLOTLY_AVAILABLE:
        st.markdown('<div id="grafica"></div>', unsafe_allow_html=True)

        sectores = {
            "Usos y Actividades": (0, 90),
            "Encuentro": (90, 180),
            "Conexiones y Accesos": (180, 270),
            "Comodidad e Imagen": (270, 360),
        }

        fig = go.Figure()

        # 1) Fondo de cada cuadrante
        for nombre_atributo, (theta_ini, theta_fin) in sectores.items():
            thetas_outer = np.linspace(theta_ini, theta_fin, 100)
            thetas_inner = np.linspace(theta_fin, theta_ini, 100)
            thetas_bg = np.concatenate([thetas_outer, thetas_inner])
            r_bg = np.concatenate(
                [
                    np.full_like(thetas_outer, radio_externo),
                    np.full_like(thetas_inner, radio_interno),
                ]
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=r_bg,
                    theta=thetas_bg,
                    mode="lines",
                    fill="toself",
                    line=dict(color="white", width=1),
                    fillcolor=color_fondo,
                    showlegend=False,
                )
            )

        # 2) Valor obtenido: área proporcional a la fracción
        for nombre_atributo, (theta_ini, theta_fin) in sectores.items():
            fraccion = valores_norm.get(nombre_atributo, 0.0)
            # r_valor = sqrt(r_int^2 + f*(r_ext^2 - r_int^2))
            radio_valor = np.sqrt(
                radio_interno**2 + fraccion * (radio_externo**2 - radio_interno**2)
            )

            thetas_outer = np.linspace(theta_ini, theta_fin, 100)
            thetas_inner = np.linspace(theta_fin, theta_ini, 100)
            thetas_val = np.concatenate([thetas_outer, thetas_inner])
            r_val = np.concatenate(
                [
                    np.full_like(thetas_outer, radio_valor),
                    np.full_like(thetas_inner, radio_interno),
                ]
            )
            fig.add_trace(
                go.Scatterpolar(
                    r=r_val,
                    theta=thetas_val,
                    mode="lines",
                    fill="toself",
                    line=dict(color="white", width=1),
                    fillcolor=palette[nombre_atributo],
                    showlegend=False,
                )
            )

        # 3) Cruz blanca recta para separar los cuadrantes
        centros_cruz = [0, 90, 180, 270]
        for centro in centros_cruz:
            fig.add_trace(
                go.Scatterpolar(
                    r=[radio_interno, radio_externo],
                    theta=[centro, centro],
                    mode="lines",
                    line=dict(color="white", width=20),
                    showlegend=False,
                )
            )

        # 4) Nombres de los atributos por fuera del anillo de datos
        r_label_outer = radio_externo * 1.4
        labels_center = {
            "Usos y Actividades": 45,
            "Encuentro": 135,
            "Conexiones y Accesos": 225,
            "Comodidad e Imagen": 315,
        }
        for nombre_atributo, theta in labels_center.items():
            fig.add_trace(
                go.Scatterpolar(
                    r=[r_label_outer],
                    theta=[theta],
                    mode="text",
                    text=[nombre_atributo],
                    textfont=dict(color="#333", size=16),
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            height=850,
            width=850,
            polar=dict(
                radialaxis=dict(
                    visible=False,
                    range=[0, radio_externo * 1.7],
                ),
                angularaxis=dict(
                    showticklabels=False,
                    ticks="",
                ),
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Guardar PNG de la gráfica en session_state (para descarga conjunta con CSV)
        try:
            # Requiere 'kaleido': pip install kaleido
            png_bytes = fig.to_image(format="png", scale=2)
            st.session_state.grafica_png_bytes = png_bytes
        except Exception as e:
            st.session_state.grafica_png_bytes = None
            st.warning(
                f"⚠️ No se pudo exportar la gráfica como PNG. "
                f"Esto es opcional y no afecta el funcionamiento de la app.\n\n"
                f"**Detalles del error:** {type(e).__name__}: {str(e)}\n\n"
                f"Para resolver esto:\n"
                f"1. Asegúrate de que 'kaleido' está instalado: `pip install kaleido`\n"
                f"2. Si estás en un servidor Linux, puede que falten dependencias del sistema. "
                f"En ese caso, puedes ignorar este error y usar la función 'Imprimir/Guardar como PDF' "
                f"que aparece más abajo en la página."
            )

        # Scroll suave hacia la gráfica cuando ya está lista
        scroll_to_here(200, key="grafica")

        # Botón para ir al análisis de resultados (SIN key duplicado)
        if st.button("Ir al análisis de resultados", key="ir_analisis_btn"):
            scroll_to_here(200, key="analisis_resultados")

    else:
        st.error(
            "La librería Plotly no está disponible en el entorno donde se ejecuta Streamlit. "
            "Instálala en ese entorno y vuelve a cargar la app."
        )
        st.code(f"{sys.executable} -m pip install plotly kaleido", language="bash")
        if _plotly_import_err:
            st.caption(f"Error de importación: {_plotly_import_err}")

    # -----------------------------------------------------
    # Interpretación de resultados (a, b, c, d)
    # -----------------------------------------------------
    st.markdown('<div id="analisis_resultados"></div>', unsafe_allow_html=True)
    st.header("Interpretación de tus resultados")

    # Helper para clasificar niveles
    def nivel_desempeno(fraccion: float) -> str:
        if fraccion >= 0.85:
            return "excelente"
        elif fraccion >= 0.65:
            return "bueno"
        elif fraccion >= 0.4:
            return "en desarrollo"
        else:
            return "crítico"

    fraccion_total = total_puntos / max_total if max_total > 0 else 0.0
    nivel_global = nivel_desempeno(fraccion_total)

    texto_global_a = ""
    interpretacion_b = {}
    texto_c_general = ""
    comentarios_c = []
    comentarios_d = []

    # a) Qué tan excelente es su lugar
    st.subheader("¿Qué tan excelente es tu lugar?")
    if nivel_global == "excelente":
        texto_global_a = (
            "En conjunto, tu lugar se encuentra en un nivel **excelente**. "
            "La mayoría de los atributos evaluados se desempeñan muy bien, lo que indica que "
            "es un espacio vivo, atractivo y con fuerte sentido de comunidad. "
            "Las personas probablemente lo reconocen como un referente del barrio y disfrutan "
            "permanecer ahí."
        )
    elif nivel_global == "bueno":
        texto_global_a = (
            "Tu lugar muestra un desempeño **bueno** en general. "
            "Hay bases sólidas en varios atributos y el espacio funciona de manera adecuada, "
            "pero aún tiene margen de mejora para convertirse en un lugar verdaderamente "
            "extraordinario. Con algunas intervenciones específicas, puede pasar de ser "
            "un buen lugar a un lugar excelente."
        )
    elif nivel_global == "en desarrollo":
        texto_global_a = (
            "Tu lugar se encuentra **en desarrollo**. "
            "Existen cualidades importantes, pero también vacíos claros en uno o varios "
            "atributos. Esto sugiere que el sitio todavía no aprovecha todo su potencial "
            "para generar encuentro, ofrecer actividades variadas, ser fácilmente accesible "
            "o proyectar una imagen cómoda y segura."
        )
    else:
        texto_global_a = (
            "Actualmente, tu lugar se encuentra en una situación **crítica**. "
            "Varios atributos presentan puntuaciones bajas, lo que puede traducirse en un "
            "espacio poco utilizado, percibido como inseguro o desconectado. "
            "Sin embargo, este diagnóstico también señala una gran oportunidad: pequeñas "
            "acciones estratégicas pueden detonar cambios significativos."
        )
    st.write(texto_global_a)

    # b) Qué significan sus resultados (por atributo)
    st.subheader("¿Qué significan tus resultados por atributo?")
    for nombre_atributo in orden_atributos:
        fr = valores_norm.get(nombre_atributo, 0.0)
        nivel = nivel_desempeno(fr)
        puntaje = subtotales[nombre_atributo]
        max_attr = atributos[nombre_atributo]["max"]

        if nombre_atributo == "Encuentro":
            if nivel == "excelente":
                texto = (
                    f"En **Encuentro** obtuviste {puntaje} de {max_attr} puntos, lo que refleja "
                    "un nivel excelente. Tu lugar facilita que las personas se vean, saluden "
                    "a vecinos, convivan con familias, niñas, niños y personas mayores, y se "
                    "sienta un fuerte orgullo y sentido de pertenencia. Es un espacio donde "
                    "la comunidad realmente se reconoce y se encuentra."
                )
            elif nivel == "bueno":
                texto = (
                    f"En **Encuentro** obtuviste {puntaje} de {max_attr} puntos. "
                    "Tu lugar funciona como un punto de reunión aceptable: hay presencia de "
                    "familias y grupos diversos, y cierta organización vecinal, aunque aún se "
                    "podría fortalecer la participación comunitaria y la sensación de orgullo "
                    "y apropiación del lugar."
                )
            elif nivel == "en desarrollo":
                texto = (
                    f"En **Encuentro** obtuviste {puntaje} de {max_attr} puntos. "
                    "El sitio ofrece algunas oportunidades de interacción, pero estas son "
                    "limitadas o esporádicas. Podría haber poca evidencia de vecinos "
                    "organizados o de actividades que integren a niñas, niños y personas "
                    "mayores, lo que reduce el apego al lugar."
                )
            else:
                texto = (
                    f"En **Encuentro** obtuviste {puntaje} de {max_attr} puntos, lo que indica "
                    "un nivel crítico. Es probable que el lugar no se use como punto de "
                    "reunión, que casi no haya interacción entre vecinos y que los grupos "
                    "demográficos (niñas, niños, mayores, familias) estén poco presentes. "
                    "Esto limita fuertemente el sentido de comunidad en el espacio."
                )

        elif nombre_atributo == "Usos y Actividades":
            if nivel == "excelente":
                texto = (
                    f"En **Usos y Actividades** obtuviste {puntaje} de {max_attr} puntos. "
                    "Tu lugar ofrece muchas opciones de actividades, las personas se quedan "
                    "tiempo, las niñas y niños se ven entretenidos y el mobiliario resulta "
                    "funcional. Además, la mezcla de comercios y servicios y la vitalidad "
                    "económica hacen que siempre haya algo que hacer."
                )
            elif nivel == "bueno":
                texto = (
                    f"En **Usos y Actividades** obtuviste {puntaje} de {max_attr} puntos. "
                    "El sitio tiene varias actividades y usos, pero todavía hay momentos o "
                    "zonas donde no pasa mucho. La oferta de mobiliario, comercio o servicios "
                    "es adecuada, aunque podría diversificarse para atraer a más personas "
                    "y prolongar su permanencia."
                )
            elif nivel == "en desarrollo":
                texto = (
                    f"En **Usos y Actividades** obtuviste {puntaje} de {max_attr} puntos. "
                    "Probablemente hay pocas opciones claras de actividad y el espacio se "
                    "percibe más como un lugar de paso que de estancia. Esto hace que el sitio "
                    "luzca vacío en ciertos momentos y que el mobiliario o los servicios no "
                    "estén aprovechados."
                )
            else:
                texto = (
                    f"En **Usos y Actividades** obtuviste {puntaje} de {max_attr} puntos, "
                    "indicando un nivel crítico. Casi no hay razones para quedarse en el lugar: "
                    "faltan actividades, servicios atractivos o mobiliario útil. "
                    "En estas condiciones, el espacio tiende a permanecer vacío y poco visible "
                    "para la comunidad."
                )

        elif nombre_atributo == "Conexiones y Accesos":
            if nivel == "excelente":
                texto = (
                    f"En **Conexiones y Accesos** obtuviste {puntaje} de {max_attr} puntos. "
                    "Tu lugar es fácil de alcanzar, caminar y recorrer: está bien conectado "
                    "con su entorno, tiene paradas de transporte público cercanas, "
                    "rampas accesibles y señalización clara. Esto facilita que muchas personas "
                    "lo usen diariamente."
                )
            elif nivel == "bueno":
                texto = (
                    f"En **Conexiones y Accesos** obtuviste {puntaje} de {max_attr} puntos. "
                    "En general, el lugar es accesible y visible, pero puede haber ciertos "
                    "tramos incómodos para caminar, falta de rampas en algunos puntos o "
                    "señalización que podría ser más clara. Aun así, la mayoría de las "
                    "personas puede llegar sin demasiadas dificultades."
                )
            elif nivel == "en desarrollo":
                texto = (
                    f"En **Conexiones y Accesos** obtuviste {puntaje} de {max_attr} puntos. "
                    "El sitio no siempre resulta fácil de alcanzar o atravesar: quizá la "
                    "ubicación no es tan conveniente, la caminabilidad es limitada o la "
                    "conexión con el transporte público es débil. Esto reduce el flujo de "
                    "personas que pueden disfrutar del lugar."
                )
            else:
                texto = (
                    f"En **Conexiones y Accesos** obtuviste {puntaje} de {max_attr} puntos, "
                    "lo que señala un nivel crítico. Es probable que llegar al lugar sea "
                    "difícil, que no existan buenas rutas peatonales ni rampas, y que la "
                    "señalización sea escasa o confusa. Todo esto hace que el sitio parezca "
                    "aislado o poco visible."
                )

        else:  # Comodidad e Imagen
            if nivel == "excelente":
                texto = (
                    f"En **Comodidad e Imagen** obtuviste {puntaje} de {max_attr} puntos. "
                    "El lugar se percibe atractivo, agradable y seguro; está limpio, bien "
                    "mantenido y cuenta con suficientes lugares cómodos para sentarse. "
                    "La gente se siente a gusto permaneciendo ahí."
                )
            elif nivel == "bueno":
                texto = (
                    f"En **Comodidad e Imagen** obtuviste {puntaje} de {max_attr} puntos. "
                    "La imagen general del lugar es positiva, aunque es posible que haya "
                    "detalles de mantenimiento, limpieza o cantidad de asientos que podrían "
                    "mejorarse para aumentar la sensación de confort y seguridad."
                )
            elif nivel == "en desarrollo":
                texto = (
                    f"En **Comodidad e Imagen** obtuviste {puntaje} de {max_attr} puntos. "
                    "Es probable que el sitio presente cierta incomodidad: pocos asientos, "
                    "áreas poco agradables o percepción de inseguridad en ciertos horarios. "
                    "La limpieza y el mantenimiento podrían no ser constantes."
                )
            else:
                texto = (
                    f"En **Comodidad e Imagen** obtuviste {puntaje} de {max_attr} puntos, "
                    "lo que indica un nivel crítico. El lugar puede percibirse sucio, "
                    "descuidado o inseguro, con escasos sitios para sentarse y poca "
                    "sensación de confort. Esto desincentiva que las personas permanezcan."
                )

        st.write(texto)
        interpretacion_b[nombre_atributo] = texto

    # c) Cómo es su lugar en general (qué puede mejorar en cada atributo)
    st.subheader("¿Cómo es tu lugar en general y qué puede mejorar?")
    texto_c_general = (
        "En conjunto, tu lugar combina el desempeño de los cuatro atributos evaluados: "
        "Encuentro, Usos y Actividades, Conexiones y Accesos, y Comodidad e Imagen. "
        "Los resultados muestran en qué dimensiones el espacio ya funciona bien y en cuáles "
        "aún hay una brecha para que las personas lo sientan verdaderamente propio, "
        "vivo, accesible y agradable."
    )
    st.write(texto_c_general)

    for nombre_atributo in orden_atributos:
        fr = valores_norm.get(nombre_atributo, 0.0)
        nivel = nivel_desempeno(fr)
        if nivel in ["en desarrollo", "crítico"]:
            if nombre_atributo == "Encuentro":
                msg = (
                    "En **Encuentro**, es importante fortalecer la presencia de vecinas y "
                    "vecinos organizados, así como de niñas, niños, personas mayores y "
                    "familias. Actividades que inviten a conocerse y saludarse pueden "
                    "transformar el lugar en un verdadero punto de reunión."
                )
            elif nombre_atributo == "Usos y Actividades":
                msg = (
                    "En **Usos y Actividades**, tu lugar necesita más motivos para que la "
                    "gente llegue y se quede: juegos para niñas y niños, mobiliario útil, "
                    "programación cultural, ferias o actividades comunitarias que doten de "
                    "vida cotidiana al espacio."
                )
            elif nombre_atributo == "Conexiones y Accesos":
                msg = (
                    "En **Conexiones y Accesos**, conviene revisar cómo se llega al lugar: "
                    "rutas peatonales seguras, cruces claros, accesibilidad universal y "
                    "vínculo con el transporte público. Mejorar estos elementos hará que "
                    "más personas lo usen."
                )
            else:
                msg = (
                    "En **Comodidad e Imagen**, mejorar la limpieza, el mantenimiento, la "
                    "iluminación y la cantidad/calidad de asientos puede cambiar por completo "
                    "la percepción del espacio, haciéndolo más atractivo y seguro."
                )
            st.write("- " + msg)
            comentarios_c.append(msg)

    # d) Cómo puede mejorar su lugar (recomendaciones)
    st.subheader("¿Cómo puedes mejorar tu lugar?")
    st.write(
        "A partir de tu diagnóstico, puedes plantear un plan de mejora gradual. "
        "Te sugerimos priorizar acciones de bajo costo y alto impacto, involucrando "
        "a la comunidad desde el inicio:"
    )

    for nombre_atributo in orden_atributos:
        fr = valores_norm.get(nombre_atributo, 0.0)
        nivel = nivel_desempeno(fr)

        if nombre_atributo == "Encuentro":
            if nivel in ["excelente", "bueno"]:
                msg = (
                    "**Encuentro**: Mantén y refuerza las actividades comunitarias que ya "
                    "funcionan (asambleas, talleres, festivales, tianguis, juegos en familia). "
                    "Documentar y celebrar estas prácticas ayuda a consolidar el orgullo y la "
                    "identidad del lugar."
                )
            else:
                msg = (
                    "**Encuentro**: Organiza actividades simples para activar el espacio, "
                    "como vecinos limpiando juntos, tardes de juegos para niñas y niños, "
                    "cine al aire libre o círculos de lectura. Estas dinámicas facilitan que "
                    "las personas se conozcan y se sientan parte de una misma comunidad."
                )

        elif nombre_atributo == "Usos y Actividades":
            if nivel in ["excelente", "bueno"]:
                msg = (
                    "**Usos y Actividades**: Diversifica la oferta actual cuidando la "
                    "rotación de actividades (culturales, deportivas, recreativas) para "
                    "mantener el interés. Involucra a comerciantes y colectivos locales para "
                    "que el lugar siga siendo un punto de referencia cotidiano."
                )
            else:
                msg = (
                    "**Usos y Actividades**: Introduce nuevos usos ligeros y reversibles: "
                    "mobiliario móvil, juegos pintados en el piso, espacios para mercados "
                    "temporales o ferias comunitarias. Observa qué actividades generan mayor "
                    "permanencia y ajusta el espacio según esa respuesta."
                )

        elif nombre_atributo == "Conexiones y Accesos":
            if nivel in ["excelente", "bueno"]:
                msg = (
                    "**Conexiones y Accesos**: Refuerza la señalización y la continuidad "
                    "peatonal para mantener la buena accesibilidad. Asegúrate de que las "
                    "rutas sigan siendo cómodas y seguras, especialmente para niñas, niños, "
                    "personas mayores y personas con discapacidad."
                )
            else:
                msg = (
                    "**Conexiones y Accesos**: Trabaja en mejorar cruces peatonales, "
                    "aceras, rampas y señalización. Acciones como pintar pasos de cebra, "
                    "agregar señalética clara o coordinar con autoridades para mejorar "
                    "el transporte pueden abrir el lugar a más personas."
                )

        else:  # Comodidad e Imagen
            if nivel in ["excelente", "bueno"]:
                msg = (
                    "**Comodidad e Imagen**: Cuida el mantenimiento continuo (limpieza, "
                    "pintura, jardinería) y la iluminación. Involucra a la comunidad en el "
                    "cuidado del espacio para mantener la percepción de seguridad y confort."
                )
            else:
                msg = (
                    "**Comodidad e Imagen**: Prioriza intervenciones visibles: limpiar, "
                    "pintar, reparar mobiliario, mejorar la iluminación y añadir más "
                    "asientos cómodos. Pequeños cambios físicos pueden transformar la "
                    "experiencia de quienes usan el lugar."
                )

        st.write("- " + msg)
        comentarios_d.append(msg)

    # -----------------------------------------------------
    # Descarga de resultados en CSV + PNG
    # -----------------------------------------------------
    st.header("Descarga de resultados")

    filas = []
    for nombre_atributo, info in atributos.items():
        for i, pregunta in enumerate(info["preguntas"]):
            valor = respuestas[nombre_atributo][i]
            filas.append(
                {
                    "Lugar": nombre_lugar,
                    "Evaluador": nombre_evaluador,
                    "FechaHoraEvaluacion": timestamp_eval,
                    "Atributo": nombre_atributo,
                    "Pregunta": pregunta,
                    "Puntuación": valor,
                }
            )

    df_resultados = pd.DataFrame(filas)
    csv = df_resultados.to_csv(index=False, encoding="utf-8-sig")

    col_csv, col_png = st.columns(2)

    with col_csv:
        st.download_button(
            label="Descargar respuestas en CSV",
            data=csv,
            file_name="place_game_respuestas.csv",
            mime="text/csv",
        )

    with col_png:
        if st.session_state.grafica_png_bytes:
            st.download_button(
                label="Descargar gráfica en PNG",
                data=st.session_state.grafica_png_bytes,
                file_name="place_game_grafica.png",
                mime="image/png",
            )
        else:
            st.write(
                "Gráfica no disponible como PNG. "
                "Asegúrate de tener instalada la librería 'kaleido' en el entorno."
            )

    # -----------------------------------------------------
    # Botón para imprimir / guardar como PDF (del lado del navegador)
    # -----------------------------------------------------
    st.subheader("Imprimir o guardar esta página como PDF")
    st.write(
        "Para guardar todo el reporte (gráfica, interpretación y resultados) como PDF, "
        "haz clic en el botón de abajo o usa el comando de impresión de tu navegador "
        "(Ctrl+P o Cmd+P) y elige la opción **Guardar como PDF**."
    )

    if st.button("Imprimir / Guardar como PDF", key="imprimir_pdf"):
        js_print = """
        <script>
        window.parent.print();
        </script>
        """
        components.html(js_print, height=0)

# ---------------------------------------------------------
# Scroll pendiente a secciones del cuestionario (sec_X)
# ---------------------------------------------------------
if isinstance(st.session_state.get("scroll_target"), int):
    idx = st.session_state.scroll_target
    scroll_to_here(200, key=f"sec_{idx}")
    st.session_state.scroll_target = None
