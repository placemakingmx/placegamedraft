import os
import csv
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitSecretNotFoundError

PLOTLY_AVAILABLE = True
PLOTLY_ERROR = None
try:
    import plotly.express as px
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
        import plotly.express as px
    except Exception as exc:
        PLOTLY_AVAILABLE = False
        PLOTLY_ERROR = exc
        px = None

MATPLOTLIB_AVAILABLE = True
MATPLOTLIB_ERROR = None
try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:
        MATPLOTLIB_AVAILABLE = False
        MATPLOTLIB_ERROR = exc
        plt = None
        font_manager = None

PYCIRCLIZE_AVAILABLE = True
PYCIRCLIZE_ERROR = None
try:
    from pycirclize import Circos
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pycirclize"])
        from pycirclize import Circos
    except Exception as exc:
        PYCIRCLIZE_AVAILABLE = False
        PYCIRCLIZE_ERROR = exc
        Circos = None

GSPREAD_AVAILABLE = True
GSPREAD_ERROR = None
try:
    import gspread
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gspread"])
        import gspread
    except Exception as exc:
        GSPREAD_AVAILABLE = False
        GSPREAD_ERROR = exc
        gspread = None

def get_value(ans, key, na_key=None, cast=float, scale=1.0):
    if na_key and ans.get(na_key):
        return None
    v = ans.get(key)
    if v is None:
        return None
    try:
        return cast(v) * scale
    except (TypeError, ValueError):
        return None


SHEET_COLUMNS = [
    "nombre_lugar",
    "nombre_evaluador",
    "programa",
    "genero_id",
    "equipo_responsable_id",
    "A1.1",
    "A1.2",
    "A1.3",
    "A1.4",
    "A1.5",
    "A1_total",
    "A2.1",
    "A2.2",
    "A2.3",
    "A2.4",
    "A2.5",
    "A2.6",
    "A2_total",
    "A3.1",
    "A3.2",
    "A3.3",
    "A3.4",
    "A3.5",
    "A3.6",
    "A3.7",
    "A3_total",
    "A4.1",
    "A4.2",
    "A4.3",
    "A4.4",
    "A4.5",
    "A4.6",
    "A4_total",
    "global_score",
]


def _safe_secrets() -> dict:
    try:
        return st.secrets
    except StreamlitSecretNotFoundError:
        return {}


def _get_sheet_config() -> tuple[str | None, str]:
    secrets = _safe_secrets()
    sheet_url = secrets.get("google_sheet_url") if "google_sheet_url" in secrets else None
    if not sheet_url:
        sheet_url = os.getenv("GOOGLE_SHEET_URL")
    sheet_tab = secrets.get("google_sheet_tab") if "google_sheet_tab" in secrets else None
    if not sheet_tab:
        sheet_tab = os.getenv("GOOGLE_SHEET_TAB", "Hoja 1")
    return sheet_url, sheet_tab


def append_to_google_sheet(row_data: dict) -> tuple[bool, str]:
    if not GSPREAD_AVAILABLE:
        detail = f" Detalle técnico: {GSPREAD_ERROR}" if GSPREAD_ERROR else ""
        return False, (
            "No se pudo cargar gspread en el entorno actual. "
            f"Instala con: {sys.executable} -m pip install gspread.{detail}"
        )
    sheet_url, sheet_tab = _get_sheet_config()
    if not sheet_url:
        return False, "Falta GOOGLE_SHEET_URL en st.secrets o variables de entorno."
    secrets = _safe_secrets()
    if "gcp_service_account" not in secrets:
        return False, "Falta gcp_service_account en st.secrets."

    client = gspread.service_account_from_dict(secrets["gcp_service_account"])
    worksheet = client.open_by_url(sheet_url).worksheet(sheet_tab)

    row = [row_data.get(col, "") for col in SHEET_COLUMNS]
    worksheet.append_row(row, value_input_option="USER_ENTERED")
    return True, "OK"


# ──────────────────────────────────────────
# 1. Ajustes de interfaz básica
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Indicadores del lugar",
    page_icon="uploads/carita.png" if os.path.exists("uploads/carita.png") else None,
    layout="centered",
)

HIDE_HEADER = """
<style>
header[data-testid="stHeader"] {display:none;}
#MainMenu, footer {visibility:hidden;}
</style>
"""
st.markdown(HIDE_HEADER, unsafe_allow_html=True)


# ──────────────────────────────────────────
# 2. Cargar CSS y logotipo
# ──────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "uploads" / "styles.css"
LOGO_PATH = BASE_DIR / "uploads" / "logo.png"

def cargar_css_local(css_path: Path) -> None:
    """Inyecta CSS local si existe."""
    if css_path.exists():
        with css_path.open("r", encoding="utf-8") as f:
            css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

cargar_css_local(CSS_PATH)

if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=170)

# ──────────────────────────────────────────
# 3. Estado y navegación
# ──────────────────────────────────────────
# Initialize core state
if "answers" not in st.session_state:
    st.session_state["answers"] = {}
if "save_pending" not in st.session_state:
    st.session_state["save_pending"] = False
if "saved_to_sheet" not in st.session_state:
    st.session_state["saved_to_sheet"] = False
if "force_reload" not in st.session_state:
    st.session_state["force_reload"] = False

if st.session_state.get("force_reload"):
    st.session_state["force_reload"] = False
    components.html("<script>window.parent.location.reload();</script>", height=0)
    st.stop()

def reset_evaluacion() -> None:
    # Clear widget state while keeping base keys we re-init below.
    keep_keys = {
        "answers",
        "save_pending",
        "saved_to_sheet",
        "force_reload",
    }
    keys_to_clear = [k for k in st.session_state.keys() if k not in keep_keys]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    st.session_state["answers"] = {}
    st.session_state["save_pending"] = False
    st.session_state["saved_to_sheet"] = False
    st.session_state["force_reload"] = True
    st.rerun()

def trigger_save_to_sheet() -> None:
    st.session_state["save_pending"] = True

def get_ans(key, default=None):
    return st.session_state["answers"].get(key, default)

def set_ans(key, value):
    st.session_state["answers"][key] = value

def _ui_key(base: str) -> str:
    # Use stable keys per question
    return base

def _clear_answer(*answer_keys: str) -> None:
    """Clear stored answers (logical values)."""
    for k in answer_keys:
        st.session_state["answers"].pop(k, None)

def _clear_widget_state(*base_keys: str) -> None:
    """Clear Streamlit widget state for keys that were built with _ui_key(base)."""
    for base in base_keys:
        st.session_state.pop(_ui_key(base), None)

def clear_branch(*base_keys: str) -> None:
    """
    Clears both the saved answer and the widget UI state
    for widgets that use base key == answer key.
    """
    _clear_answer(*base_keys)
    _clear_widget_state(*base_keys)

def to_float_or_none(x):
    """Safe float conversion for result parsing."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

# Helpers de widgets (radio, checkbox, slider, etc.)
def radio_answer(key, label, options, labels_map):
    prev = get_ans(key)
    idx = options.index(prev) if prev in options else 0
    ui_key = _ui_key(key)
    val = st.radio(label, options, index=idx, format_func=lambda x: labels_map[x], key=ui_key)
    set_ans(key, val)
    return val

def checkbox_answer(key, label):
    prev = get_ans(key, False)
    ui_key = _ui_key(key)
    val = st.checkbox(label, value=prev, key=ui_key)
    set_ans(key, val)
    return val

def slider_answer(key, label, min_value, max_value, step=1):
    prev = get_ans(key, min_value)
    ui_key = _ui_key(key)
    val = st.slider(label, min_value, max_value, prev, step=step, key=ui_key)
    set_ans(key, val)
    return val

def number_answer(key, label, min_value=0.0, max_value=100.0, step=1.0):
    prev = get_ans(key, min_value)
    ui_key = _ui_key(key)
    val = st.number_input(label, min_value=min_value, max_value=max_value, step=step, value=prev, key=ui_key)
    set_ans(key, val)
    return val

def selectbox_answer(key, label, options, default_index=0):
    prev = get_ans(key)
    idx = options.index(prev) if prev in options else default_index
    ui_key = _ui_key(key)
    val = st.selectbox(label, options, index=idx, key=ui_key)
    set_ans(key, val)
    return val

# =========================================================
# 1. CONFIGURACIÓN DE PROGRAMAS Y PROYECTOS
# =========================================================

# Etiquetas visibles en A0.0.1 → IDs internos de programa
PROGRAM_LABELS = {
    "1) Otro programa": "OTRO",
    "2) Programa FIESTA": "FIESTA",
    "3A) Programa LAPIS (dentro de un espacio privado como una escuela)": "LAPIS_PRIV",
    "3B) Programa LAPIS (en un espacio público)": "LAPIS_PUB",
    "4) Programa LAPIS +": "LAPIS_PLUS",
    "5) Programa PINTA TU CANCHA": "PINTA_CANCHA",
    "6) Programa Canchas con Placemaking": "CANCHAS_PM",
    "7) Programa Relacionamiento comunitario": "REL_COM",
    "8) Programa Backing International Small Restaurants": "BACKING",
    "9) Programa Menú del día": "MENU_DIA",
    "10) Programa Seguridad / Higiene y Empoderamiento (SHE)": "SHE",
    "11) Programa Adaptaciones basadas en ecosistemas en comunidades amigables para personas mayores": "ECO_ADAPT",
    "12) Salud Digna": "SALUD_DIGNA",
    "13) Placemaking Camp": "PM_CAMP",
}

COMMON_SECTION_WEIGHTS = {
    "Encuentro": 0.25,
    "Conexiones": 0.25,
    "Comodidad": 0.25,
    "Usos": 0.25,
}

PROGRAM_CONFIG = {
    # 1) Otro programa → no altera nada
    "OTRO": {
        "indicator_weights": {},
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 2) Programa FIESTA
    "FIESTA": {
        "indicator_weights": {
            "A2": {  # Conexiones y accesos
                "A2.1": 0.7,  # Partición modal: quitar un poco
                "A2.3": 0.7,  # Partición modal: quitar un poco
                "A2.6": 1.5,  # Accesibilidad 1a infancia/cuidadores: más peso
            },
            "A4": {  # Usos y actividades
                "A4.4": 0.3,  # Actividad económica: poco peso para casi todos
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 3A) LAPIS (espacio privado)
    "LAPIS_PRIV": {
        "indicator_weights": {
            "A2": {  # Conexiones
                "A2.1": 0.7,
                "A2.3": 0.7,
                "A2.6": 1.5,
            },
            "A3": {  # Comodidad e imagen
                "A3.3": 1.5,
            },
            "A4": {  # Usos
                "A4.4": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 3B) LAPIS (espacio público)
    "LAPIS_PUB": {
        "indicator_weights": {
            "A1": {  # Encuentro
                "A1.2": 0.0,  # Redes ciudadanas sin peso
                "A1.5": 0.0,  # Uso nocturno sin peso
            },
            "A2": {  # Conexiones
                "A2.1": 0.7,
                "A2.3": 0.7,
                "A2.6": 1.5,
            },
            "A3": {  # Comodidad e imagen
                "A3.3": 1.5,
            },
            "A4": {  # Usos
                "A4.4": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 4) LAPIS +
    "LAPIS_PLUS": {
        "indicator_weights": {
            "A2": {  # Conexiones
                "A2.1": 0.7,
                "A2.3": 0.7,
                "A2.6": 1.5,
            },
            "A3": {  # Comodidad
                "A3.3": 1.5,
            },
            "A4": {  # Usos
                "A4.4": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 5) PINTA TU CANCHA
    "PINTA_CANCHA": {
        "indicator_weights": {
            "A2": {
                "A2.1": 0.7,
                "A2.3": 0.7,
                "A2.6": 1.5,
            },
            "A3": {
                "A3.3": 0.5,  # Quitar peso para CANCHAS
            },
            "A4": {
                "A4.4": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 6) Canchas con Placemaking
    "CANCHAS_PM": {
        "indicator_weights": {
            "A2": {
                "A2.1": 0.7,
                "A2.3": 0.7,
                "A2.6": 1.5,
            },
            "A3": {
                "A3.3": 0.5,
            },
            "A4": {
                "A4.4": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 7) Relacionamiento comunitario
    "REL_COM": {
        "indicator_weights": {
            "A2": {
                "A2.1": 0.7,
                "A2.2": 1.5,
                "A2.3": 0.7,
                "A2.6": 1.5,
            },
            "A4": {
                "A4.4": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 8) Backing International Small Restaurants
    "BACKING": {
        "indicator_weights": {
            "A2": {
                "A2.1": 0.7,
                "A2.3": 0.7,
            },
            "A4": {
                "A4.4": 2.5,  # Actividad económica muy importante
                "A4.5": 0.3,  # Diversidad de actividades poco peso
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 9) Menú del día
    "MENU_DIA": {
        "indicator_weights": {
            "A2": {
                "A2.1": 0.7,
                "A2.3": 0.7,
            },
            "A4": {
                "A4.4": 2.5,
                "A4.5": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 10) Seguridad / Higiene y Empoderamiento (SHE)
    "SHE": {
        "indicator_weights": {
            "A1": {  # Encuentro
                "A1.1": 0.0,
                "A1.2": 0.0,
                "A1.5": 2.0,
            },
            "A2": {  # Conexiones
                "A2.1": 0.7,
                "A2.2": 1.5,
                "A2.3": 0.7,
            },
            "A3": {  # Comodidad
                "A3.3": 0.5,
            },
            "A4": {  # Usos
                "A4.4": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 11) Adaptaciones basadas en ecosistemas...
    "ECO_ADAPT": {
        "indicator_weights": {
            "A2": {
                "A2.1": 0.7,
                "A2.3": 0.7,
            },
            "A4": {
                "A4.4": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 12) Salud Digna
    "SALUD_DIGNA": {
        "indicator_weights": {
            "A2": {
                "A2.1": 0.7,
                "A2.3": 0.7,
            },
            "A4": {
                "A4.4": 0.3,
            },
        },
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
    # 13) Placemaking Camp (sin reglas específicas todavía)
    "PM_CAMP": {
        "indicator_weights": {},
        "section_weights": COMMON_SECTION_WEIGHTS,
    },
}


# ----------------- PROYECTOS (tabla completa) -----------------
PROJECTS = [
    # ========= PROGRAMA FIESTA =========
    {
        "id": 1,
        "program_id": "FIESTA",
        "estado": "Jalisco",
        "municipio": "La Trinidad",
        "tipologia": "LQC",
        "proyecto": "Parque la Trinidad",
        "gmaps_url": "https://maps.app.goo.gl/7tyfLyHdgmczTJPdA",
        "lat": 20.67758819149736,
        "lon": -102.48229836255611,
    },
    {
        "id": 2,
        "program_id": "FIESTA",
        "estado": "Guanajuato",
        "municipio": "San Cristóbal",
        "tipologia": "LQC",
        "proyecto": "Plaza de San Cristóbal (Plaza Principal)",
        "gmaps_url": "https://maps.app.goo.gl/KyohDU6JoizgGE1WA",
        "lat": 20.978865299355252,
        "lon": -101.69765085704643,
    },
    {
        "id": 3,
        "program_id": "FIESTA",
        "estado": "Michoacán",
        "municipio": "Caurio de Guadalupe",
        "tipologia": "LQC",
        "proyecto": "La cancha de Caurio de Guadalupe",
        "gmaps_url": "https://maps.app.goo.gl/SAvoQJb2ShuJ5g2A7",
        "lat": 19.922164345221834,
        "lon": -101.86273344907545,
    },
    {
        "id": 4,
        "program_id": "FIESTA",
        "estado": "Michoacán",
        "municipio": "Peribán de Ramos",
        "tipologia": "LQC",
        "proyecto": "Unidad deportiva La Joyita",
        "gmaps_url": "https://maps.app.goo.gl/ytVdJFSemiH6yaMCA",
        "lat": 19.522671920787687,
        "lon": -102.41130209140428,
    },
    {
        "id": 5,
        "program_id": "FIESTA",
        "estado": "Michoacán",
        "municipio": "Tanaquillo",
        "tipologia": "LQC",
        "proyecto": "Parque comunal de Tanaquillo",
        "gmaps_url": "https://maps.app.goo.gl/QVTXEwig9MaEJc4V9",
        "lat": 19.85106474145374,
        "lon": -102.09620273558226,
    },
    {
        "id": 6,
        "program_id": "FIESTA",
        "estado": "Michoacán",
        "municipio": "Los Reyes de Salgado",
        "tipologia": "LQC",
        "proyecto": "Parque La Zarzamora",
        "gmaps_url": "https://maps.app.goo.gl/V7CcXvLQZqH8mjSx6",
        "lat": 19.58730737371672,
        "lon": -102.47765887976014,
    },
    {
        "id": 7,
        "program_id": "FIESTA",
        "estado": "Jalisco",
        "municipio": "Ahualulco del Mercado",
        "tipologia": "LQC",
        "proyecto": "Parque el Mezquite",
        "gmaps_url": "https://maps.app.goo.gl/KSmTDTJybYgFwHmA6",
        "lat": 20.706216897862074,
        "lon": -103.98875333494489,
    },
    {
        "id": 8,
        "program_id": "FIESTA",
        "estado": "Jalisco",
        "municipio": "Tapalpa",
        "tipologia": "LQC",
        "proyecto": "Cancha La Loma",
        "gmaps_url": "https://maps.app.goo.gl/Zxp2A8fam1wv6Zgn6",
        "lat": 19.944177509085915,
        "lon": -103.76290809180293,
    },
    {
        "id": 9,
        "program_id": "FIESTA",
        "estado": "Jalisco",
        "municipio": "Jocotepec",
        "tipologia": "LQC",
        "proyecto": "Unidad deportiva Norte",
        "gmaps_url": "https://maps.app.goo.gl/8HKJWbnwqe1ofE2t8",
        "lat": 20.29167155234112,
        "lon": -103.43547002571187,
    },
    {
        "id": 10,
        "program_id": "FIESTA",
        "estado": "Jalisco",
        "municipio": "Ciudad Guzmán",
        "tipologia": "LQC",
        "proyecto": "Unidad deportiva Salvador Aguilar",
        "gmaps_url": "https://maps.app.goo.gl/CbNhZioeavyKxd487",
        "lat": 19.69440906176885,
        "lon": -103.47786827791103,
    },

    # ========= PROGRAMA LAPIS PRIV =========
    {
        "id": 11,
        "program_id": "LAPIS_PRIV",
        "estado": "CDMX",
        "municipio": "Cuajimalpa de Morelos",
        "tipologia": "LQC",
        "proyecto": "CENDI Sor Juana Inés de la Cruz",
        "gmaps_url": "https://maps.app.goo.gl/dds98GBSZuLkoZiX8",
        "lat": 19.36933476307679,
        "lon": -99.29371400674661,
    },
    {
        "id": 12,
        "program_id": "LAPIS_PRIV",
        "estado": "CDMX",
        "municipio": "Cuajimalpa de Morelos",
        "tipologia": "LQC",
        "proyecto": "CENDI Ignacio Manuel Altamirano",
        "gmaps_url": "https://maps.app.goo.gl/7V2p2gQddK4ZQ4vg6",
        "lat": 19.373422924669455,
        "lon": -99.28862963743131,
    },
    {
        "id": 13,
        "program_id": "LAPIS_PRIV",
        "estado": "CDMX",
        "municipio": "Álvaro Obregón",
        "tipologia": "LQC",
        "proyecto": "CACI Jalalpa",
        "gmaps_url": "https://maps.app.goo.gl/VwucQvRYNurgC9YRA",
        "lat": 19.371714411087062,
        "lon": -99.23983702208896,
    },
    {
        "id": 14,
        "program_id": "LAPIS_PRIV",
        "estado": "Baja California",
        "municipio": "Mexicali",
        "tipologia": "LQC",
        "proyecto": "Jardín de niños Centenario de Mexicali",
        "gmaps_url": "https://maps.app.goo.gl/vobXibBCJF9mhieU6",
        "lat": 32.58339122284644,
        "lon": -115.35286336678451,
    },
    {
        "id": 15,
        "program_id": "LAPIS_PRIV",
        "estado": "Baja California",
        "municipio": "Tijuana",
        "tipologia": "LQC",
        "proyecto": "Jardín de niños Torres de Agua Caliente",
        "gmaps_url": "https://maps.app.goo.gl/TxAQsJ1LvY3fDL3N8",
        "lat": 32.38906212745669,
        "lon": -116.95276303558221,
    },
    {
        "id": 26,
        "program_id": "LAPIS_PRIV",
        "estado": "Sonora",
        "municipio": "Huatabampo",
        "tipologia": "LQC",
        "proyecto": "Jardín de niños Ignacio Altamirano",
        "gmaps_url": "https://maps.app.goo.gl/ZgCNqBhLsrwrTuaQ9",
        "lat": 26.82067201553576,
        "lon": -109.64003618278342,
    },
    {
        "id": 27,
        "program_id": "LAPIS_PRIV",
        "estado": "Nayarit",
        "municipio": "San Blas",
        "tipologia": "LQC",
        "proyecto": "Jardín de niños Guillermo Prieto",
        "gmaps_url": "https://maps.app.goo.gl/nQWG8qyHhJvtwVQ97",
        "lat": 21.548031988283093,
        "lon": -105.28795345227265,
    },
    {
        "id": 28,
        "program_id": "LAPIS_PRIV",
        "estado": "Baja California Sur",
        "municipio": "Loreto",
        "tipologia": "LQC",
        "proyecto": "Jardín de niños Juan Escutia",
        "gmaps_url": "https://maps.app.goo.gl/rBrWwEidqjMSiTxD8",
        "lat": 26.011718734951366,
        "lon": -111.34287930954382,
    },

    # ========= PROGRAMA LAPIS PUB =========
    {
        "id": 16,
        "program_id": "LAPIS_PUB",
        "estado": "Michoacán",
        "municipio": "Morelia",
        "tipologia": "LQC",
        "proyecto": "Cancha Benito Rocha",
        "gmaps_url": "https://maps.app.goo.gl/3W9uyLByNVWxkxHk7",
        "lat": 19.674096387701905,
        "lon": -101.22656145520986,
    },
    {
        "id": 17,
        "program_id": "LAPIS_PUB",
        "estado": "Michoacán",
        "municipio": "Morelia",
        "tipologia": "LQC",
        "proyecto": "Unidad Deportiva Morelos INDECO",
        "gmaps_url": "https://maps.app.goo.gl/6ts5ynq6MEyLX7By9",
        "lat": 19.682437844648817,
        "lon": -101.22804780368567,
    },
    {
        "id": 18,
        "program_id": "LAPIS_PUB",
        "estado": "Michoacán",
        "municipio": "Morelia",
        "tipologia": "LQC",
        "proyecto": "Canchas Clavijero",
        "gmaps_url": "https://maps.app.goo.gl/QCA2Zd7Suy33StQ79",
        "lat": 19.698513736399658,
        "lon": -101.14371032518154,
    },
    {
        "id": 19,
        "program_id": "LAPIS_PUB",
        "estado": "Querétaro",
        "municipio": "Querétaro",
        "tipologia": "LQC",
        "proyecto": "Parque Cuitláhuac",
        "gmaps_url": "https://maps.app.goo.gl/Dr1Ah6KeMynNXn8w6",
        "lat": 20.551455713447798,
        "lon": -100.37897750674664,
    },
    {
        "id": 20,
        "program_id": "LAPIS_PUB",
        "estado": "Querétaro",
        "municipio": "San Juan del Río",
        "tipologia": "LQC",
        "proyecto": "Parque El Capricho",
        "gmaps_url": "https://maps.app.goo.gl/VqDCct3GUBTa5AsR9",
        "lat": 20.37507733791507,
        "lon": -99.95207793558221,
    },
    {
        "id": 21,
        "program_id": "LAPIS_PUB",
        "estado": "Querétaro",
        "municipio": "San Juan del Río",
        "tipologia": "LQC",
        "proyecto": "Parque Las Haciendas",
        "gmaps_url": "https://maps.app.goo.gl/jSSgZuhVmiAwf3t68",
        "lat": 20.383567741688765,
        "lon": -99.96653316441781,
    },
    {
        "id": 22,
        "program_id": "LAPIS_PRIV",
        "estado": "Querétaro",
        "municipio": "San Juan del Río",
        "tipologia": "LQC",
        "proyecto": "Parque Santa Cruz Escandón",
        "gmaps_url": "https://maps.app.goo.gl/R9YGE3AmGxSoqTFJ8",
        "lat": 20.416892934101128,
        "lon": -99.95517010859572,
    },
    {
        "id": 23,
        "program_id": "LAPIS_PUB",
        "estado": "Estado de México",
        "municipio": "Toluca",
        "tipologia": "LQC",
        "proyecto": "Unidad Deportiva el Olimpo",
        "gmaps_url": "https://maps.app.goo.gl/txWWhR4kFkaqMasGA",
        "lat": 19.30312975767844,
        "lon": -99.59399735822055,
    },
    {
        "id": 24,
        "program_id": "LAPIS_PUB",
        "estado": "Estado de México",
        "municipio": "Toluca",
        "tipologia": "LQC",
        "proyecto": "Parque Los Sauces",
        "gmaps_url": "https://maps.app.goo.gl/xcFZB2rE19ECf1Kz6",
        "lat": 19.35377902220411,
        "lon": -99.59451590674662,
    },
    {
        "id": 25,
        "program_id": "LAPIS_PUB",
        "estado": "Tamaulipas",
        "municipio": "Tampico",
        "tipologia": "LQC",
        "proyecto": "Parque Tiburón",
        "gmaps_url": "https://maps.app.goo.gl/txWWhR4kFkaqMasGA",
        "lat": 22.277860523114356,
        "lon": -97.88085284100397,
    },
    {
        "id": 29,
        "program_id": "LAPIS_PUB",
        "estado": "Oaxaca",
        "municipio": "Bajos de Chila",
        "tipologia": "LQC",
        "proyecto": "Delegación Las Tres Palmas",
        "gmaps_url": "https://maps.app.goo.gl/96oBQWwVufAgusu46",
        "lat": 15.912190560514201,
        "lon": -97.13118100859572,
    },

    # ========= PROGRAMA LAPIS + (PUB) =========
    {
        "id": 30,
        "program_id": "LAPIS_PLUS",
        "estado": "Guanajuato",
        "municipio": "León",
        "tipologia": "LQC",
        "proyecto": "Centro de Desarrollo Familiar",
        "gmaps_url": "https://maps.app.goo.gl/KATo2rpaXmYNLUmy6",
        "lat": 21.105974904723443,
        "lon": -101.68656799325339,
    },
    {
        "id": 31,
        "program_id": "LAPIS_PLUS",
        "estado": "Yucatán",
        "municipio": "Mérida",
        "tipologia": "LQC",
        "proyecto": "Parque Jardines, DIF Parque Arena",
        "gmaps_url": "https://maps.app.goo.gl/ckFVde78uWgX9ScZA",
        "lat": 20.918118337984566,
        "lon": -89.67738063428249,
    },
    {
        "id": 32,
        "program_id": "LAPIS_PLUS",
        "estado": "Querétaro",
        "municipio": "Querétaro",
        "tipologia": "LQC",
        "proyecto": "Parque San Pablo",
        "gmaps_url": "https://maps.app.goo.gl/gQsqJtP11SSUB6Fr9",
        "lat": 20.614751481398027,
        "lon": -100.41796963558221,
    },

    # ========= PROGRAMA PINTA TU CANCHA =========
    {
        "id": 33,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Iztapalapa",
        "tipologia": "Canchas",
        "proyecto": "PILARES Central de Abasto",
        "gmaps_url": "https://maps.app.goo.gl/AW6NpT1PpNkyHHAa8",
        "lat": 19.373872226228144,
        "lon": -99.09928039140543,
    },
    {
        "id": 34,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Venustiano Carranza",
        "tipologia": "Canchas",
        "proyecto": "Centro Deportivo Felipe 'Tibio' Muñóz",
        "gmaps_url": "https://maps.app.goo.gl/oZx4an7eZsQ3q59X8",
        "lat": 19.430426744739346,
        "lon": -99.05862436604134,
    },
    {
        "id": 35,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Tlalpan",
        "tipologia": "Canchas",
        "proyecto": "PILARES Tequistlatecos",
        "gmaps_url": "https://maps.app.goo.gl/mSBd3E6f6XkbH2bc7",
        "lat": 19.273137771890266,
        "lon": -99.19037533569273,
    },
    {
        "id": 36,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Miguel Hidalgo",
        "tipologia": "Canchas",
        "proyecto": "Módulo Deportivo Plan Sexenal",
        "gmaps_url": "https://maps.app.goo.gl/AqhD2h2RLov18zP49",
        "lat": 19.455232486439805,
        "lon": -99.17195409753462,
    },
    {
        "id": 37,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Benito Juárez",
        "tipologia": "Canchas",
        "proyecto": "Parque Álamos",
        "gmaps_url": "https://maps.app.goo.gl/1SUiAvSH3iqB7jTA6",
        "lat": 19.39845809792983,
        "lon": -99.14230068388436,
    },
    {
        "id": 38,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Iztapalapa",
        "tipologia": "Canchas",
        "proyecto": "Utopía Libertad",
        "gmaps_url": "https://maps.app.goo.gl/6dHajdU2g72fPBmd7",
        "lat": 19.324835610736905,
        "lon": -99.06546168263094,
    },
    {
        "id": 39,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Tláhuac",
        "tipologia": "Canchas",
        "proyecto": "Deportivo Juan Palomo Martínez",
        "gmaps_url": "https://maps.app.goo.gl/odMkwu7nC7JFfjtA8",
        "lat": 19.265885695405736,
        "lon": -99.00232749930322,
    },
    {
        "id": 40,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Xochimilco",
        "tipologia": "Canchas",
        "proyecto": "PILARES Ahualapa",
        "gmaps_url": "https://maps.app.goo.gl/YQTAhyKannEXQjUq5",
        "lat": 19.247185030419995,
        "lon": -99.06926392771148,
    },
    {
        "id": 41,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Azcapotzalco",
        "tipologia": "Canchas",
        "proyecto": "Pilares Coltongo",
        "gmaps_url": "https://maps.app.goo.gl/LjKwxKzgYstzges68",
        "lat": 19.4819650397926,
        "lon": -99.15311065122081,
    },
    {
        "id": 42,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Gustavo A. Madero",
        "tipologia": "Canchas",
        "proyecto": "Pilares Richard Wagner",
        "gmaps_url": "https://maps.app.goo.gl/GDvKQLASN2yYq38A8",
        "lat": 19.465787964372673,
        "lon": -99.12881391285796,
    },
    {
        "id": 43,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Coyoacán",
        "tipologia": "Canchas",
        "proyecto": "Parque Cantera",
        "gmaps_url": "https://maps.app.goo.gl/dQVuwvdpByepBAYG8",
        "lat": 19.310534936719364,
        "lon": -99.16497164338703,
    },
    {
        "id": 44,
        "program_id": "PINTA_CANCHA",
        "estado": "Jalisco",
        "municipio": "Zapopan",
        "tipologia": "Canchas",
        "proyecto": "Unidad Deportiva Las Margaritas",
        "gmaps_url": "https://maps.app.goo.gl/GuqqhfvRy2sdw8Bw7",
        "lat": 20.741617792952557,
        "lon": -103.41890976445559,
    },
    {
        "id": 45,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Magdalena Contreras",
        "tipologia": "Canchas",
        "proyecto": "Cancha Tierra Unida",
        "gmaps_url": "https://maps.app.goo.gl/cvov3hZwguvaDgUU6",
        "lat": 19.303604478528765,
        "lon": -99.27019112362419,
    },
    {
        "id": 46,
        "program_id": "PINTA_CANCHA",
        "estado": "Nuevo León",
        "municipio": "Apodaca",
        "tipologia": "Canchas",
        "proyecto": "Centro Comunitario Santa Fe",
        "gmaps_url": "https://maps.app.goo.gl/uworipwUqFXrvoVPA",
        "lat": 25.726300545470448,
        "lon": -100.17340339557092,
    },
    {
        "id": 47,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Milpa Alta",
        "tipologia": "Canchas",
        "proyecto": "Módulo Deportivo Tecpallo",
        "gmaps_url": "https://maps.app.goo.gl/tWYCdTBF175BZZWM7",
        "lat": 19.187258449171157,
        "lon": -98.99300733294854,
    },
    {
        "id": 48,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Iztacalco",
        "tipologia": "Canchas",
        "proyecto": "Pilares La Fortaleza",
        "gmaps_url": "https://maps.app.goo.gl/RGt2HhKcARqkoC9f6",
        "lat": 19.387936098518434,
        "lon": -99.09147247865148,
    },
    {
        "id": 49,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Cuajimalpa de Morelos",
        "tipologia": "Canchas",
        "proyecto": "Deportivo Morelos",
        "gmaps_url": "https://maps.app.goo.gl/m6aRpf4BG8kL44uY6",
        "lat": 19.36420775262693,
        "lon": -99.28557127998937,
    },
    {
        "id": 50,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Álvaro Obregón",
        "tipologia": "Canchas",
        "proyecto": "Pilares La Araña",
        "gmaps_url": "https://maps.app.goo.gl/VcW9HG299YNi21mS9",
        "lat": 19.36283301859252,
        "lon": -99.23985397594383,
    },
    {
        "id": 51,
        "program_id": "PINTA_CANCHA",
        "estado": "Baja California",
        "municipio": "Tijuana",
        "tipologia": "Canchas",
        "proyecto": "Cancha Playas de Tijuana",
        "gmaps_url": "https://maps.app.goo.gl/JFFTrbq5mgLB2Ewc9",
        "lat": 32.514021079275736,
        "lon": -117.11157969401623,
    },
    {
        "id": 52,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Cuauhtémoc",
        "tipologia": "Canchas",
        "proyecto": "Pilares Atlampa",
        "gmaps_url": "https://maps.app.goo.gl/U5xp3kmakUUfVnth9",
        "lat": 19.46056065349821,
        "lon": -99.16011491461214,
    },
    {
        "id": 53,
        "program_id": "PINTA_CANCHA",
        "estado": "CDMX",
        "municipio": "Cuajimalpa de Morelos",
        "tipologia": "Canchas",
        "proyecto": "Deportivo Maracaná",
        "gmaps_url": "https://maps.app.goo.gl/EQ8jPZ7bUm7wpCCz9",
        "lat": 19.37511578720077,
        "lon": -99.28939501428543,
    },

    # ========= PROGRAMA CANCHAS CON PLACEMAKING =========
    {
        "id": 54,
        "program_id": "CANCHAS_PM",
        "estado": "Jalisco",
        "municipio": "Ocotlán",
        "tipologia": "Canchas",
        "proyecto": "Canchas de la Secundaria Técnica 42 de Ocotlán",
        "gmaps_url": "https://maps.app.goo.gl/Nx7TwEiXJYKB2nhb7",
        "lat": 20.362990465163577,
        "lon": -102.77876111860053,
    },

    # ========= PROGRAMA RELACIONAMIENTO COMUNITARIO =========
    {
        "id": 55,
        "program_id": "REL_COM",
        "estado": "Zacatecas",
        "municipio": "Mazapil",
        "tipologia": "Relaciones Comunitarias",
        "proyecto": "Parque Raíces de Mazapil",
        "gmaps_url": "https://maps.app.goo.gl/s7Raqx21bejC6cnLA",
        "lat": 24.641814131244995,
        "lon": -101.55442982208875,
    },
    {
        "id": 56,
        "program_id": "REL_COM",
        "estado": "Colima",
        "municipio": "Alzada",
        "tipologia": "Relaciones Comunitarias",
        "proyecto": "Proyecto de mejoramiento urbano",
        "gmaps_url": "https://maps.app.goo.gl/WzxGzEaLVjyiajXP7",
        "lat": 19.258367991196106,
        "lon": -103.52801836441773,
    },

    # ========= PROGRAMA BACKING =========
    {
        "id": 57,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Cuauhtémoc",
        "tipologia": "Restaurantes",
        "proyecto": "Freims",
        "gmaps_url": "https://maps.app.goo.gl/eJqCAi8FbyGKVnXo8",
        "lat": 19.41474122828503,
        "lon": -99.16959017791129,
    },
    {
        "id": 58,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Cuauhtémoc",
        "tipologia": "Restaurantes",
        "proyecto": "Guau Tap",
        "gmaps_url": "https://maps.app.goo.gl/3uxbwxJTvYZPSYwR9",
        "lat": 19.437834666619374,
        "lon": -99.16222684907581,
    },
    {
        "id": 59,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Miguel Hidalgo",
        "tipologia": "Restaurantes",
        "proyecto": "Lovecraft Café",
        "gmaps_url": "https://maps.app.goo.gl/EC6q2rLwSUxiKT77",
        "lat": 19.412137938112917,
        "lon": -99.18220936441773,
    },
    {
        "id": 60,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Benito Juárez",
        "tipologia": "Restaurantes",
        "proyecto": "Mictlan Antojitos Veganos",
        "gmaps_url": "https://maps.app.goo.gl/DCgh2W2EP8NNsS1q8",
        "lat": 19.391113082729117,
        "lon": -99.15615062024034,
    },
    {
        "id": 61,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Cuauhtémoc",
        "tipologia": "Restaurantes",
        "proyecto": "Q' Pedro Pablo",
        "gmaps_url": "https://maps.app.goo.gl/3u5mvfXCnqWqxPDU9",
        "lat": 19.42082752468557,
        "lon": -99.16164227975965,
    },
    {
        "id": 62,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Tlalpan",
        "tipologia": "Restaurantes",
        "proyecto": "Artesano De Barrio",
        "gmaps_url": "https://maps.app.goo.gl/J5XkjK4yw3ERfKPd9",
        "lat": 19.28691955703127,
        "lon": -99.18202929140483,
    },
    {
        "id": 63,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Cuauhtémoc",
        "tipologia": "Restaurantes",
        "proyecto": "Gracias Madre Taquería Vegana",
        "gmaps_url": "https://maps.app.goo.gl/qdcV9NcKCZFKuLJU6",
        "lat": 19.419872106800575,
        "lon": -99.15722336441775,
    },
    {
        "id": 64,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Benito Juárez",
        "tipologia": "Restaurantes",
        "proyecto": "La Caravana",
        "gmaps_url": "https://maps.app.goo.gl/NmuAj2g1FyYDUugGA",
        "lat": 19.377850144066098,
        "lon": -99.17555434737268,
    },
    {
        "id": 65,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Cuauhtémoc",
        "tipologia": "Restaurantes",
        "proyecto": "La Llorona Cantina",
        "gmaps_url": "https://maps.app.goo.gl/AvPE7vbLe346mLMY7",
        "lat": 19.416153071407123,
        "lon": -99.16953486441774,
    },
    {
        "id": 66,
        "program_id": "BACKING",
        "estado": "CDMX",
        "municipio": "Cuauhtémoc",
        "tipologia": "Restaurantes",
        "proyecto": "Somos Voces Cafetería / Centro Mexicano de Libros",
        "gmaps_url": "https://maps.app.goo.gl/UvEEQ1VnMdSuserr5",
        "lat": 19.427342530913034,
        "lon": -99.16310359325324,
    },

    # ========= PROGRAMA MENÚ DEL DÍA =========
    {
        "id": 67,
        "program_id": "MENU_DIA",
        "estado": "CDMX",
        "municipio": "Benito Juárez",
        "tipologia": "Capacitaciones Fonda",
        "proyecto": "Cocina Malenita",
        "gmaps_url": "https://maps.app.goo.gl/Jm2rt1133da9Tk2Q8",
        "lat": 19.395327574947963,
        "lon": -99.16599484110445,
    },
    {
        "id": 68,
        "program_id": "MENU_DIA",
        "estado": "CDMX",
        "municipio": "Cuauhtémoc",
        "tipologia": "Capacitaciones Fonda",
        "proyecto": "Cocina Godínez",
        "gmaps_url": "https://maps.app.goo.gl/XzyoKKJzuvAi8jX97",
        "lat": 19.409188635998774,
        "lon": -99.17320273558225,
    },

    # ========= PROGRAMA SHE =========
    {
        "id": 69,
        "program_id": "SHE",
        "estado": "Estado de México",
        "municipio": "Cuautitlán",
        "tipologia": "Relaciones Comunitarias",
        "proyecto": "CEDIS Cuautitlán",
        "gmaps_url": "https://maps.app.goo.gl/gP17qakgnf5zomySA",
        "lat": 19.659653092889947,
        "lon": -99.18705970674678,
    },
    {
        "id": 70,
        "program_id": "SHE",
        "estado": "Estado de México",
        "municipio": "Ecatepec",
        "tipologia": "Relaciones Comunitarias",
        "proyecto": "Planta Essity Ecatepec",
        "gmaps_url": "https://maps.app.goo.gl/XdjuUX1mDmuuGqct7",
        "lat": 19.58265193843687,
        "lon": -99.03295576187666,
    },
    {
        "id": 71,
        "program_id": "SHE",
        "estado": "Hidalgo",
        "municipio": "Ciudad Sahagún",
        "tipologia": "Relaciones Comunitarias",
        "proyecto": "Planta Essity Ciudad Sahagún",
        "gmaps_url": "https://maps.app.goo.gl/Y9z7nwv692aD5ke1A",
        "lat": 19.735525578044168,
        "lon": -98.58566922356962,
    },
    {
        "id": 72,
        "program_id": "SHE",
        "estado": "Nuevo León",
        "municipio": "Monterrey",
        "tipologia": "Relaciones Comunitarias",
        "proyecto": "Planta Essity Monterrey",
        "gmaps_url": "https://maps.app.goo.gl/jHYTTkzgAPSESVFZ7",
        "lat": 25.73066581768624,
        "lon": -100.29139162024032,
    },
    {
        "id": 73,
        "program_id": "SHE",
        "estado": "Michoacán",
        "municipio": "Uruapan",
        "tipologia": "Relaciones Comunitarias",
        "proyecto": "Planta Essity Uruapan",
        "gmaps_url": "https://maps.app.goo.gl/9omMRyboeMgwPeaS9",
        "lat": 19.424424505273336,
        "lon": -102.01796665031189,
    },

    # ========= PROGRAMA ECO_ADAPT =========
    {
        "id": 74,
        "program_id": "ECO_ADAPT",
        "estado": "CDMX",
        "municipio": "Gustavo A. Madero",
        "tipologia": "Relaciones Comunitarias",
        "proyecto": "Huerto del comedor comunitario Nuevos Horizontes",
        "gmaps_url": "https://maps.app.goo.gl/fcYQcK99nbZQLJ8z6",
        "lat": 19.48388061062029,
        "lon": -99.09150619257203,
    },
    {
        "id": 75,
        "program_id": "ECO_ADAPT",
        "estado": "CDMX",
        "municipio": "Gustavo A. Madero",
        "tipologia": "Relaciones Comunitarias",
        "proyecto": "Comedor comunitario Nuevos Horizontes",
        "gmaps_url": "https://maps.app.goo.gl/PHLV61Dg9H6jfVFNA",
        "lat": 19.48409417505207,
        "lon": -99.09154277975966,
    },

    # ========= PROGRAMA SALUD DIGNA =========
    {
        "id": 76,
        "program_id": "SALUD_DIGNA",
        "estado": "Sinaloa",
        "municipio": "Culiacán",
        "tipologia": "Plan maestro",
        "proyecto": "Plan de Placemaking Campus Salud Digna (The Place Institute)",
        "gmaps_url": "https://maps.app.goo.gl/WUvgfzAG7DQkBDG46",
        "lat": 24.74744785052024,
        "lon": -107.43894788845039,
    },
]

PROJECTS_BY_PROGRAM = defaultdict(list)
for p in PROJECTS:
    PROJECTS_BY_PROGRAM[p["program_id"]].append(p)


# =========================================================
# 1.b FUNCIONES AUXILIARES DE PONDERACIÓN
# =========================================================
def nz(x, default=0.0):
    """Devuelve default si x es None, para evitar errores."""
    return default if x is None else x


def impute_missing_with_penalty(
    indicators: dict[str, float | None],
    penalty: float = 40.0,
) -> dict[str, float]:
    """
    Imputa valores None usando el promedio de los demás indicadores menos una penalización fija.
    """
    available = {k: v for k, v in indicators.items() if v is not None}
    missing_keys = [k for k, v in indicators.items() if v is None]

    if not missing_keys:
        return {k: float(v) for k, v in available.items()}

    if not available:
        return {k: 20.0 for k in indicators.keys()}

    avg_available = sum(available.values()) / len(available)
    imputed_value = avg_available - penalty
    imputed_value = max(20.0, imputed_value)

    result = {}
    for key, value in indicators.items():
        if value is not None:
            result[key] = float(value)
        else:
            result[key] = imputed_value

    return result


def get_indicator_weights(program_id: str, attribute_id: str, indicators: dict) -> dict:
    """Pesos de indicadores de un atributo para un programa dado."""
    program_cfg = PROGRAM_CONFIG.get(program_id, {})
    attr_weights = program_cfg.get("indicator_weights", {}).get(attribute_id, {})
    if not attr_weights:
        return {ind_id: 1.0 for ind_id in indicators.keys()}
    weights = {}
    for ind_id in indicators.keys():
        weights[ind_id] = attr_weights.get(ind_id, 1.0)
    return weights


def get_section_weights(program_id: str) -> dict:
    """Pesos de secciones (Encuentro, Conexiones, etc.) para un programa."""
    program_cfg = PROGRAM_CONFIG.get(program_id, {})
    section_weights = program_cfg.get("section_weights", COMMON_SECTION_WEIGHTS)
    total = sum(section_weights.values())
    if total == 0:
        return section_weights
    return {k: v / total for k, v in section_weights.items()}


def compute_attribute_total(program_id: str, attribute_id: str, indicators: dict) -> float:
    """
    MEJORADO: Imputa valores None usando promedio de hermanos - 40 puntos.
    Promedio ponderado de indicadores (0–100) de un atributo.
    """
    if not indicators:
        return 0.0

    indicators_imputed = impute_missing_with_penalty(indicators, penalty=40.0)

    weights = get_indicator_weights(program_id, attribute_id, indicators_imputed)
    num = 0.0
    den = 0.0
    for ind_id, value in indicators_imputed.items():
        w = weights.get(ind_id, 1.0)
        num += w * value
        den += w

    return num / den if den > 0 else 0.0


def compute_section_scores(A1_total: float, A2_total: float, A3_total: float, A4_total: float) -> dict:
    """Atributos"""
    return {
        "Encuentro": A1_total,
        "Conexiones": A2_total,
        "Comodidad": A3_total,
        "Usos": A4_total,
    }

def compute_global_score(
    program_id: str,
    section_scores: dict,
    sostenible_score: float | None = None,
    sostenible_weight: float = 0.15,
    conmemorativo_score: float | None = None,
    conmemorativo_weight: float = 0.10,
) -> float:
    """Score global ponderado por sección según el programa, con ajustes opcionales por intangibles."""
    if not section_scores:
        return 0.0
    section_weights = get_section_weights(program_id)
    num = 0.0
    den = 0.0
    for section, value in section_scores.items():
        if value is None:
            continue
        w = section_weights.get(section, 0.0)
        num += w * value
        den += w
    base_score = num / den if den > 0 else 0.0

    has_sostenible = sostenible_score is not None
    has_conmemorativo = conmemorativo_score is not None
    if not has_sostenible and not has_conmemorativo:
        return base_score

    ws = max(0.0, min(1.0, sostenible_weight if has_sostenible else 0.0))
    wc = max(0.0, min(1.0, conmemorativo_weight if has_conmemorativo else 0.0))

    total_intangible_weight = ws + wc
    if total_intangible_weight > 1.0:
        ws = ws / total_intangible_weight
        wc = wc / total_intangible_weight

    wb = 1.0 - ws - wc
    result = wb * base_score
    if has_sostenible:
        result += ws * float(sostenible_score)
    if has_conmemorativo:
        result += wc * float(conmemorativo_score)
    return result


# =========================================================
# 2. FUNCIONES DE MAPEOS BÁSICOS
# =========================================================
def score_1_3_100_50_10(x):
    if x == 1:
        return 100.0
    elif x == 2:
        return 50.0
    elif x == 3:
        return 10.0
    return None


def score_1_3_10_50_100(x):
    if x == 1:
        return 10.0
    elif x == 2:
        return 50.0
    elif x == 3:
        return 100.0
    return None


def score_1_4_100_75_50_25(x):
    if x == 1:
        return 100.0
    elif x == 2:
        return 75.0
    elif x == 3:
        return 50.0
    elif x == 4:
        return 25.0
    return None


def score_1_4_100_60_20_10(x):
    if x == 1:
        return 100.0
    elif x == 2:
        return 60.0
    elif x == 3:
        return 20.0
    elif x == 4:
        return 10.0
    return None


def score_estado(a):
    """Estado físico del lugar: 1→100, 2→75, 3→50, 4→25."""
    return score_1_4_100_75_50_25(a)


def score_A1_5_1(a15_1):
    """Score S5.1: 1→100, 2→10, 3→5."""
    if a15_1 == 1:
        return 100.0
    elif a15_1 == 2:
        return 10.0
    elif a15_1 == 3:
        return 5.0
    return None


# =========================================================
# 3. CÁLCULO DE INDICADORES E INTANGIBLES A1 (ENCUENTRO)
# =========================================================
def calc_A1_1(a11_1, a11_2):
    if a11_1 is None:
        return None
    if a11_2 is None:
        return a11_1
    if a11_1 >= a11_2:
        return 0.85 * a11_1 + 0.15 * a11_2
    else:
        return 0.60 * a11_1 + 0.40 * a11_2


def score_A1_2_1(a12_1):
    if a12_1 == 1:
        return 10.0
    elif a12_1 == 2:
        return 50.0
    elif a12_1 == 3:
        return 100.0
    return None


def score_A1_2_2(a12_1, a12_2_1, a12_2_2):
    if a12_1 != 1:
        return score_1_3_100_50_10(a12_2_1)
    else:
        return score_1_4_100_75_50_25(a12_2_2)


def score_A1_2_3(a12_3):
    if a12_3 == 1:
        return 100.0
    elif a12_3 == 2:
        return 1.0
    elif a12_3 == 3:
        return 5.0
    return None


def calc_A1_2(a12_1, a12_2_1, a12_2_2, a12_3):
    S21 = score_A1_2_1(a12_1)
    S22 = score_A1_2_2(a12_1, a12_2_1, a12_2_2)
    S23 = score_A1_2_3(a12_3)
    if None in (S21, S22, S23):
        return None
    return 0.75 * S21 + 0.10 * S22 + 0.15 * S23


def calc_A1_3(a12_1, a13_1, a12_2_1, a12_2_2):
    # Caso con redes
    if a12_1 != 1:
        estado = score_1_4_100_75_50_25(a13_1)
        cuidado_redes = score_1_3_100_50_10(a12_2_1)

        if estado is None and cuidado_redes is None:
            return None
        if estado is None:
            return cuidado_redes
        if cuidado_redes is None:
            return estado

        return 0.65 * estado + 0.35 * cuidado_redes
    # Caso sin redes: se toma estado físico y se penaliza 30%
    base = score_1_4_100_75_50_25(a12_2_2)
    if base is None:
        return None
    return base * 0.7


def calc_A1_4(A1_3, a14_1):
    S41 = score_estado(a14_1)
    if A1_3 is None or S41 is None:
        return None
    return 0.5 * A1_3 + 0.5 * S41


def calc_A1_5(a12_3, a15_1, a15_2):
    S51 = score_A1_5_1(a15_1)
    S52 = score_1_4_100_60_20_10(a15_2)
    S23 = score_A1_2_3(a12_3)
    if a15_1 == 1:
        return S51
    if S52 is None or S23 is None:
        return None
    return 0.5 * S52 + S23


def calc_intangibles_A1(A1_1, A1_2, A1_3, A1_4, A1_5):
    """Calcula intangibles con promedios simples. Los None se manejan con imputación posterior."""
    A1_1 = nz(A1_1)
    A1_2 = nz(A1_2)
    A1_3 = nz(A1_3)
    A1_4 = nz(A1_4)
    A1_5 = nz(A1_5)

    return {
        "Diversidad": A1_1,
        "Cuidado": 0.34 * A1_1 + 0.33 * A1_2 + 0.33 * A1_4,
        "Comunidad": 0.5 * A1_2 + 0.5 * A1_3,
        "Compartido": A1_2,
        "Símbolos": 0.5 * A1_2 + 0.5 * A1_3,
        "Orgullo": A1_3,
        "Amigable": A1_5,
        "Interactivo": A1_5,
    }


# =========================================================
# 4. CÁLCULO DE INDICADORES E INTANGIBLES A2
# =========================================================
def calc_A2_1(a_walk, a_bike, a_pt, a_car, a21_2):
    if any(v is not None and v > 0 for v in [a_walk, a_bike, a_pt, a_car]):
        walk = nz(a_walk)
        bike = nz(a_bike)
        pt = nz(a_pt)
        car = nz(a_car)
        active = walk + bike
        if active > 50:
            return 100.0
        elif pt > 50:
            return 80.0
        elif car > 50:
            return 50.0
        total = active + pt + car
        if total <= 0:
            return None
        score = (1.0 * active + 0.8 * pt + 0.5 * car) / max(total, 1) * 100.0
        return score
    else:
        if a21_2 == 1:
            return 50.0
        elif a21_2 == 2:
            return 60.0
        elif a21_2 == 3:
            return 90.0
        elif a21_2 == 4:
            return 100.0
        return None


def calc_A2_2(a22_1, a22_2, a22_3):
    S221 = score_1_3_100_50_10(a22_1)
    S222 = score_1_3_10_50_100(a22_2) if a22_2 is not None else None
    if a22_3 == 1:
        S223 = 100.0
    elif a22_3 == 2:
        S223 = 50.0
    else:
        S223 = None

    S221 = nz(S221)
    S222 = nz(S222)
    S223 = nz(S223)

    if S221 == 0 and S222 == 0 and S223 == 0:
        return None

    return 0.5 * S221 + 0.2 * S222 + 0.3 * S223


def calc_A2_3(a23_1):
    return score_1_4_100_75_50_25(a23_1)


def calc_A2_4(a24_1, a24_2):
    v1 = a24_1
    v2 = a24_2
    if v1 is None and v2 is None:
        return None
    if v1 is None:
        return v2
    if v2 is None:
        return v1
    return 0.5 * v1 + 0.5 * v2


def calc_A2_5(a25_1, a25_2, a25_3):
    S251 = score_1_3_100_50_10(a25_1)
    S252 = a25_2
    S253 = score_1_3_100_50_10(a25_3)

    # Si todos son None, retornar None
    if S251 is None and S253 is None and S252 is None:
        return None

    # Si S252 es None, usar solo S251 y S253
    if S252 is None:
        if S251 is None or S253 is None:
            return None
        return 0.75 * S251 + 0.25 * S253

    # Si alguno de S251 o S253 es None, retornar None
    if S251 is None or S253 is None:
        return None

    return 0.5 * S251 + 0.3 * S252 + 0.2 * S253


def calc_A2_6(a26_1_p, a26_2, a26_2_1):
    S261 = score_1_3_100_50_10(a26_1_p)
    if S261 is None:
        return None
    if a26_2 != 1:
        return S261
    S262 = score_1_3_100_50_10(a26_2_1)
    if S262 is None:
        return S261
    return 0.65 * S261 + 0.35 * S262


def calc_intangibles_A2(A2_1, A2_2, A2_3, A2_4, A2_5, A2_6, A3_4):
    """Calcula intangibles con promedios simples. Los None se manejan con imputación posterior."""
    A2_1 = nz(A2_1)
    A2_2 = nz(A2_2)
    A2_3 = nz(A2_3)
    A2_4 = nz(A2_4)
    A2_5 = nz(A2_5)
    A2_6 = nz(A2_6)
    A3_4 = nz(A3_4)

    return {
        "Cercano": 0.5 * A2_1 + 0.5 * A2_2,
        "Conectado": 0.5 * A2_1 + 0.5 * A2_2,
        "Conveniente": 0.34 * A2_2 + 0.33 * A2_3 + 0.33 * A2_4,
        "Accesible\n(movilidad reducida)": 0.5 * A2_4 + 0.5 * A2_5,
        "Accesible (primera\ninfancia y cuidadores)": A2_6,
        "Transitable": 0.5 * A3_4 + 0.25 * A2_4 + 0.25 * A2_5,
    }


# =========================================================
# 5. CÁLCULO DE INDICADORES E INTANGIBLES A3
# =========================================================
def calc_A3_1(a31_1, a14_1):
    S311 = a31_1
    S312 = score_estado(a14_1)
    if S311 is None or S312 is None:
        return None
    return 0.6 * S311 + 0.4 * S312


def calc_A3_2(a32_1, a32_2, a14_1, A0_3):
    S321 = score_1_3_100_50_10(a32_1)
    if A0_3 == 1:
        S322 = score_1_3_100_50_10(a32_2)
        if S321 is None or S322 is None:
            return None
        return 0.55 * S321 + 0.45 * S322
    else:
        return score_estado(a14_1)


def calc_A3_3(a33_1, a33_2):
    S331 = score_1_3_100_50_10(a33_1)
    S332 = score_1_3_100_50_10(a33_2)
    if S331 is None or S332 is None:
        return None
    return 0.65 * S331 + 0.35 * S332


def calc_A3_4(a34_1):
    return score_1_3_100_50_10(a34_1)


def calc_A3_5(a35_1):
    return score_1_3_100_50_10(a35_1)


def calc_A3_6(a36_1, a36_2):
    if a36_1 is None or a36_2 is None:
        return None
    return 0.6 * a36_1 + 0.4 * a36_2


def calc_A3_7(a37_1):
    return score_1_3_100_50_10(a37_1)


def calc_intangibles_A3(A3_1, A3_2, A3_3, A3_4, A3_5, A3_6, A3_7):
    """Calcula intangibles con promedios simples. Los None se manejan con imputación posterior."""
    A3_1 = nz(A3_1)
    A3_2 = nz(A3_2)
    A3_3 = nz(A3_3)
    A3_4 = nz(A3_4)
    A3_5 = nz(A3_5)
    A3_6 = nz(A3_6)
    A3_7 = nz(A3_7)

    return {
        "Limpio": A3_1,
        "Seguro": A3_1,
        "Sentable": 0.34 * A3_2 + 0.33 * A3_3 + 0.33 * A3_5,
        "Agradable": 0.25 * A3_2 + 0.25 * A3_3 + 0.25 * A3_6 + 0.25 * A3_7,
        "Verde": 0.5 * A3_3 + 0.5 * A3_6,
        "Caminable": A3_4,
        "Resiliencia climática": A3_6,
    }


# =========================================================
# 6. CÁLCULO DE INDICADORES E INTANGIBLES A4
# =========================================================
def calc_A4_1(a41_1, a41_before, a41_after, a41_2):
    if a41_1 == 1:
        b = a41_before
        a = a41_after
        if (b is None or b <= 0) and (a is None or a <= 0):
            return 0.0
        if b is None or b <= 0:
            if a is not None and a > 0:
                return 100.0
            else:
                return 0.0
        if a is None:
            return 0.0
        g = (a - b) / b * 100.0
        if g <= 0:
            return 0.0
        elif g >= 100:
            return 100.0
        else:
            return g
    elif a41_1 == 2:
        return score_1_3_100_50_10(a41_2)
    return None


def calc_A4_2(a42_1):
    return score_1_3_100_50_10(a42_1)


def score_A4_3_1(x):
    if x == 1:
        return 100.0
    elif x == 2:
        return 75.0
    elif x == 3:
        return 50.0
    elif x == 4:
        return 1.0
    return None


def score_A4_3_2(x):
    if x == 1:
        return 100.0
    elif x == 2:
        return 75.0
    elif x == 3:
        return 50.0
    elif x == 4:
        return 1.0
    return None


def calc_A4_3(a43_1, a43_2):
    S431 = score_A4_3_1(a43_1)
    S432 = score_A4_3_2(a43_2)
    if S431 is None or S432 is None:
        return None
    return 0.6 * S431 + 0.4 * S432


def score_A4_4_1(x):
    return score_1_3_100_50_10(x)


def score_A4_4_2(x):
    return score_1_3_100_50_10(x)


def calc_A4_5(a45_1):
    if a45_1 is None:
        return None
    n = max(0, min(10, a45_1))
    return 10.0 * n


def calc_A4_6(S461, S462, A1_1):
    if S461 is None or S462 is None or A1_1 is None:
        return None
    return 0.34 * S461 + 0.33 * S462 + 0.33 * A1_1


def calc_A4_4(A0_3, a44_1, a44_2, A4_1, A4_2, A4_3, A4_5, A4_6):
    if A0_3 == 1:
        # Rama enlace/responsable
        S441 = score_A4_4_1(a44_1) if a44_1 is not None else None
        S442 = score_A4_4_2(a44_2) if a44_2 is not None else None

        if S441 is None and S442 is None:
            return None
        if S441 is None:
            return S442
        if S442 is None:
            return S441

        return 0.4 * S441 + 0.6 * S442
    else:
        vals = [v for v in [A4_1, A4_2, A4_3, A4_5, A4_6] if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)


def calc_intangibles_A4(A4_1, A4_2, A4_3, A4_4, A4_5, A4_6, A1_2, A1_3, A3_6, A3_agradable):
    """Calcula intangibles A4. Los None se manejan con imputación posterior."""
    A4_1 = nz(A4_1)
    A4_2 = nz(A4_2)
    A4_3 = nz(A4_3)
    A4_4 = nz(A4_4)
    A4_5 = nz(A4_5)
    A4_6 = nz(A4_6)
    A1_2 = nz(A1_2)
    A1_3 = nz(A1_3)
    A3_6 = nz(A3_6)
    A3_agradable = nz(A3_agradable)

    return {
        "Dinámico": 0.34 * A4_1 + 0.33 * A4_5 + 0.33 * A4_6,
        "Especial": 0.5 * A4_1 + 0.5 * A4_2,
        "Real": 0.5 * A4_2 + 0.5 * A4_3,
        "Útil": 0.5 * A4_2 + 0.5 * A4_3,
        "Local": 0.33 * A4_4 + 0.34 * A4_5 + 0.33 * A4_6,
        "Sostenible": 0.40 * A1_2 + 0.30 * A1_3 + 0.30 * A3_6,
        "Conmemorativo": 0.34 * A4_2 + 0.33 * A4_6 + 0.33 * A3_agradable,
        "Pertenencia": A4_6,
    }

# =========================================================
# 7. HELPER PARA MOSTRAR INFO GEOGRÁFICA
# =========================================================
def mostrar_info_geografica():
    answers = st.session_state.get("answers", {})
    nombre = answers.get("nombre_lugar") or answers.get("project_nombre")
    estado = answers.get("a0_estado") or answers.get("project_estado")
    municipio = answers.get("a0_municipio") or answers.get("project_municipio")
    lat = answers.get("project_lat")
    lon = answers.get("project_lon")
    gmaps_url = answers.get("project_gmaps_url")

    if not (nombre or lat or lon or gmaps_url):
        return

    st.markdown("### Ubicación del proyecto")
    if nombre:
        st.write(f"**Lugar / proyecto:** {nombre}")
    if estado or municipio:
        loc = ""
        if municipio:
            loc += municipio
        if estado:
            loc += f", {estado}" if loc else estado
        st.write(f"**Localización:** {loc}")
    if lat is not None and lon is not None:
        st.write(f"**Coordenadas:** `{lat}, {lon}`")
    if gmaps_url:
        st.markdown(f"[Ver en Google Maps]({gmaps_url})")


def format_score_with_na(value: float | None) -> str:
    """
    Formatea un score indicando claramente si es NA.

    Args:
        value: Score de 0-100 o None

    Returns:
        String formateado: "85.3" o "Sin datos"
    """
    if value is None:
        return "Sin datos"
    return f"{value:.1f}"


# =========================================================
# 8. PÁGINAS (PASO A PASO)
# =========================================================
st.title("Diagrama de Lugar")
# ANCLA AL INICIO DE LA APP
st.markdown("<div id='top-of-page'></div>", unsafe_allow_html=True)


def render_floating_back_to_top() -> None:
    """Botón flotante fijo a la derecha para volver arriba sin disparar reruns."""
    components.html(
        """
        <script>
        (function () {
            const BTN_ID = 'floating-tab-selector-btn';
            const doc = window.parent.document;

            // Limpia instancias previas (incluyendo versiones antiguas sin id estable)
            const legacy = doc.querySelectorAll(
                '#floating-tab-selector-btn, #tab-selector-singleton-btn, [data-floating-tab-selector="1"]'
            );
            legacy.forEach((el) => el.remove());

            const oldAnchors = Array.from(doc.querySelectorAll('a[href="#tab_selector"], a[href="#top-of-page"]'));
            oldAnchors
                .filter((el) => (el.textContent || '').trim() === 'Volver al selector de pestañas')
                .forEach((el) => el.remove());

            const btn = doc.createElement('a');
            btn.id = BTN_ID;
            btn.setAttribute('data-floating-tab-selector', '1');
            btn.href = '#tab_selector';
            btn.textContent = 'Volver al selector de pestañas';
            btn.style.position = 'fixed';
            btn.style.right = '20px';
            btn.style.bottom = '20px';
            btn.style.zIndex = '9999';
            btn.style.backgroundColor = 'var(--space)';
            btn.style.border = '1px solid var(--space)';
            btn.style.color = 'var(--white)';
            btn.style.borderRadius = '0.5rem';
            btn.style.padding = '0.45rem 0.9rem';
            btn.style.textDecoration = 'none';
            btn.style.display = 'inline-block';

            doc.body.appendChild(btn);
        })();
        </script>
        """,
        height=0,
    )

def pagina_antes():
    answers = st.session_state.setdefault("answers", {})

    st.markdown('<div id="header_antes"></div>', unsafe_allow_html=True)
    st.header("Antes de empezar")

    # ===========================
    # A0.0.1 - Programa a evaluar
    # ===========================
    st.subheader("Programa y proyecto")

    default_label = "1) Otro programa"
    current_label = None
    if "program_id" in answers:
        for label, pid in PROGRAM_LABELS.items():
            if pid == answers["program_id"]:
                current_label = label
                break

    selected_program_label = st.selectbox(
        "¿Qué programa quieres evaluar?",
        list(PROGRAM_LABELS.keys()),
        index=list(PROGRAM_LABELS.keys()).index(current_label or default_label),
        key="a001_program_label",
    )
    program_id = PROGRAM_LABELS[selected_program_label]
    set_ans("program_id", program_id)

    # ===========================================
    # Selección de proyecto según el programa
    # ===========================================
    proyectos_programa = PROJECTS_BY_PROGRAM.get(program_id, [])
    project_options = []
    project_map = {}

    if proyectos_programa:
        project_options.append("Otro lugar de este programa")
        for p in proyectos_programa:
            label = f'{p["id"]}. {p["proyecto"]} – {p["municipio"]}, {p["estado"]}'
            project_options.append(label)
            project_map[label] = p

        current_project_label = None
        if "project_id" in answers:
            for label, p in project_map.items():
                if p["id"] == answers["project_id"]:
                    current_project_label = label
                    break

        selected_project_label = st.selectbox(
            "Selecciona el proyecto / lugar (opcional)",
            project_options,
            index=project_options.index(current_project_label)
            if current_project_label in project_options
            else 0,
            key="project_select",
        )

        if selected_project_label != "Otro lugar de este programa":
            p = project_map[selected_project_label]
            set_ans("project_id", p["id"])
            set_ans("project_nombre", p["proyecto"])
            set_ans("project_estado", p["estado"])
            set_ans("project_municipio", p["municipio"])
            set_ans("project_tipologia", p["tipologia"])
            set_ans("project_gmaps_url", p["gmaps_url"])
            set_ans("project_lat", p["lat"])
            set_ans("project_lon", p["lon"])
            # Prefill evaluation fields with selected project info
            st.session_state["nombre_lugar_input"] = p["proyecto"]
            st.session_state["a0_estado"] = p["estado"]
            st.session_state["a0_municipio"] = p["municipio"]
            set_ans("nombre_lugar", p["proyecto"])
            set_ans("a0_estado", p["estado"])
            set_ans("a0_municipio", p["municipio"])

            st.markdown("### Información del proyecto seleccionado")
            st.write(f"**Estado:** {p['estado']}")
            st.write(f"**Ciudad/Municipio/Alcaldía:** {p['municipio']}")
            st.write(f"**Tipología:** {p['tipologia']}")
            st.write(f"**Nombre del proyecto:** {p['proyecto']}")
            st.write(f"**Coordenadas:** {p['lat']}, {p['lon']}")
            if p["gmaps_url"]:
                st.markdown(f"[Ver en Google Maps]({p['gmaps_url']})")
        else:
            # Limpia project_id si eligen "otro lugar"
            answers.pop("project_id", None)
    else:
        st.info(
        "Este programa no tiene proyectos pre-cargados. "
        "Puedes capturar el lugar manualmente."
        )

    st.markdown("---")
    st.subheader("Datos de la evaluación")

    # Campos A0.x (nombre del lugar, evaluador, estado, municipio)
    col_lugar, col_eval = st.columns(2)

    nombre_lugar_default = (
        answers.get("project_nombre") or answers.get("nombre_lugar") or ""
    )
    with col_lugar:
        if "nombre_lugar_input" in st.session_state:
            nombre_lugar = st.text_input(
                "Nombre del lugar",
                placeholder="Ej. Parque México, Reggio Emilia, etc.",
                key="nombre_lugar_input",
            )
        else:
            nombre_lugar = st.text_input(
                "Nombre del lugar",
                value=nombre_lugar_default,
                placeholder="Ej. Parque México, Reggio Emilia, etc.",
                key="nombre_lugar_input",
            )
        set_ans("nombre_lugar", nombre_lugar)

    with col_eval:
        nombre_eval = st.text_input(
            "Nombre de quien evalúa",
            value=answers.get("nombre_evaluador", ""),
            placeholder="Ej. Fred Kent, Jane Jacobs, etc.",
            key="nombre_evaluador_input",
        )
        set_ans("nombre_evaluador", nombre_eval)

    col_estado, col_municipio = st.columns(2)
    with col_estado:
        estado_default = answers.get("project_estado") or answers.get("a0_estado") or ""
        if "a0_estado" in st.session_state:
            estado = st.text_input(
                "Estado",
                key="a0_estado",
            )
        else:
            estado = st.text_input(
                "Estado",
                value=estado_default,
                key="a0_estado",
            )
        set_ans("a0_estado", estado)
    with col_municipio:
        mun_default = (
            answers.get("project_municipio") or answers.get("a0_municipio") or ""
        )
        if "a0_municipio" in st.session_state:
            municipio = st.text_input(
                "Ciudad / Municipio / Alcaldía",
                key="a0_municipio",
            )
        else:
            municipio = st.text_input(
                "Ciudad / Municipio / Alcaldía",
                value=mun_default,
                key="a0_municipio",
            )
        set_ans("a0_municipio", municipio)

    st.markdown("---")
    st.subheader("Información inicial")

    # Género
    radio_answer(
        "A0_1",
        "¿Con qué género te identificas?",
        options=[1, 2, 3],
        labels_map={
            1: "Mujer",
            2: "Hombre",
            3: "Otro / Prefiero no decir",
        },
    )

    # Equipo responsable
    radio_answer(
        "A0_3",
        "¿Eres el enlace del lugar?",
        options=[1, 2],
        labels_map={
            1: "Sí",
            2: "No",
        },
    )


def pagina_A1():
    st.markdown('<div id="header_encuentro"></div>', unsafe_allow_html=True)
    st.header("Encuentro")

    # Mostrar ubicación y coordenadas antes de la calculadora INEGI
    mostrar_info_geografica()

    # ---------------------- DIVERSIDAD DEMOGRÁFICA ----------------------
    st.markdown("**Diversidad demográfica**")
    slider_answer(
        "a11_1",
        "¿Qué porcentaje las personas que habitualmente utilizan el lugar son mujeres, niñas, niños y personas mayores?",
        0,
        100,
        1,
    )
    checkbox_answer("a11_1_na", "No lo sé / No hay datos")

    st.markdown(
        "Con la siguiente calculadora, ingresa el porcentaje de mujeres, niñas, niños "
        "y personas adultas mayores a 500m a la redonda del lugar:"
    )
    st.link_button(
    "Calculadora indicadores INEGI",
    "https://pmm-calculadora-indicadores.streamlit.app/",
    )

    number_answer(
        "a11_2",
        "Porcentaje de mujeres, niñas, niños, adolescentes "
        "y personas adultas mayores en un radio de 500m (0-10, según datos del INEGI)",
        0.0,
        10.0,
        0.1,
    )
    checkbox_answer("a11_2_na", "No lo sé / No hay datos")

    # ---------------------- REDES CIUDADANAS ----------------------
    st.markdown("**Redes ciudadanas**")
    a12_1 = radio_answer(
        "a12_1",
        "¿Existen grupos que se organicen para utilizar el lugar? (ej. torneos de fut, "
        "grupos de mamás, scouts, etc.)",
        options=[1, 2, 3],
        labels_map={
            1: "No existe ninguno",
            2: "Hay por lo menos uno",
            3: "Existen tres o más grupos",
        },
    )

    redes_sin_grupos_container = st.empty()
    redes_con_grupos_container = st.empty()
    if a12_1 == 1:
        # NO hay grupos organizados → solo se pregunta a12_2_2
        clear_branch("a12_2_1")
        redes_con_grupos_container.empty()
        with redes_sin_grupos_container.container():
            radio_answer(
                "a12_2_2",
                "¿Cómo valorarías el estado físico del lugar?",
                options=[1, 2, 3, 4],
                labels_map={
                    1: "Muy bien cuidado, parece nuevo",
                    2: "En buenas condiciones",
                    3: "Descuidado",
                    4: "En muy malas condiciones",
                },
            )
    else:
        # SÍ hay grupos organizados → solo se pregunta a12_2_1
        clear_branch("a12_2_2")
        redes_sin_grupos_container.empty()
        with redes_con_grupos_container.container():
            radio_answer(
                "a12_2_1",
                "¿Se encargan estos grupos de cuidar el lugar?",
                options=[1, 2, 3],
                labels_map={
                    1: "Sí",
                    2: "Ocasionalmente",
                    3: "No",
                },
            )

    radio_answer(
        "a12_3",
        "¿Se crearon o fortalecieron redes ciudadanas que utilicen el lugar a partir de la intervención?",
        options=[1, 2, 3],
        labels_map={1: "Sí", 2: "No", 3: "No se sabe"},
    )

    # ---------------------- VOLUNTARIADO / CUIDADO DEL LUGAR ----------------------
    cuidado_a1_container = st.empty()
    if a12_1 == 1:
        # Si NO hay redes: usamos el mismo valor de estado físico (a12_2_2) como proxy
        a12_2_2_val = get_ans("a12_2_2")
        set_ans("a13_1", a12_2_2_val)
        set_ans("a14_1", a12_2_2_val)
        cuidado_a1_container.empty()

        # Importante: limpiar posibles widgets no visibles
        clear_branch("a13_1")  # a13_1 NO se captura con widget en esta rama
        # (Pero dejamos el valor en answers, así que lo re-seteamos)
        set_ans("a13_1", a12_2_2_val)

    else:
        with cuidado_a1_container.container():
            st.markdown("**Cuidado**")

            # En esta rama, a13_1 se captura por widget y a14_1 se deriva
            a13_1_val = radio_answer(
                "a13_1",
                "¿Cómo valorarías el estado físico del lugar?",
                options=[1, 2, 3, 4],
                labels_map={
                    1: "Muy bien cuidado, parece nuevo",
                    2: "En buenas condiciones",
                    3: "Descuidado",
                    4: "En muy malas condiciones",
                },
            )
        set_ans("a14_1", a13_1_val)

    # ---------------------- USO NOCTURNO ----------------------
    st.markdown("**Uso nocturno**")
    uso_nocturno_grupos_container = st.empty()
    uso_nocturno_horario_container = st.empty()

    def _render_a15_2():
        radio_answer(
            "a15_2",
            "¿Crees que el lugar se utiliza únicamente durante el día, o hay algunas personas usándolo después de las 18:00?",
            options=[1, 2, 3, 4],
            labels_map={
                1: "Sí, siempre hay gente usando el lugar",
                2: "Hay muy pocas personas cuando oscurece",
                3: "No hay nadie después de las 18:00",
                4: "No lo sé",
            },
        )

    if a12_1 == 1:
        # Sin grupos: mostrar solo pregunta general de uso nocturno.
        clear_branch("a15_1")
        set_ans("a15_1", None)
        uso_nocturno_grupos_container.empty()
        with uso_nocturno_horario_container.container():
            _render_a15_2()
    else:
        with uso_nocturno_grupos_container.container():
            a15_1 = radio_answer(
                "a15_1",
                "¿Alguno de estos grupos lo utilizan habitualmente por las tardes/noches (después de las 18:00)?",
                options=[1, 2, 3],
                labels_map={
                    1: "Sí",
                    2: "No",
                    3: "No se sabe",
                },
            )

        if a15_1 == 1:
            # Si respondió "Sí", no mostrar pregunta general.
            clear_branch("a15_2")
            set_ans("a15_2", 1)  # lógica interna: "siempre hay gente"
            uso_nocturno_horario_container.empty()
        elif a15_1 in (2, 3):
            with uso_nocturno_horario_container.container():
                _render_a15_2()
        else:
            # Caso defensivo: si a15_1 es None
            clear_branch("a15_2")
            set_ans("a15_2", 1)
            uso_nocturno_horario_container.empty()

def pagina_A2():
    st.markdown('<div id="header_conexiones"></div>', unsafe_allow_html=True)
    st.header("Conexiones y Accesos")
    mostrar_info_geografica()

    # ---------- Modos de transporte ----------
    a21_use_percent = radio_answer(
        "a21_use_percent",
        "¿Conoces los porcentajes por modo de transporte que utilizan las personas para llegar al lugar?",
        options=[1, 2],
        labels_map={1: "Sí", 2: "No"},
    )

    modos_percent_container = st.empty()
    modos_majority_container = st.empty()
    if a21_use_percent == 1:
        # Rama con porcentajes → a21_2 no aplica
        clear_branch("a21_2")
        modos_majority_container.empty()

        with modos_percent_container.container():
            slider_answer("a21_1_walk", "Porcentaje de personas que llega caminando", 0, 100)
            slider_answer("a21_1_bike", "Porcentaje que llega en bicicleta", 0, 100)
            slider_answer("a21_1_pt", "Porcentaje que llega en transporte público", 0, 100)
            slider_answer("a21_1_car", "Porcentaje que llega en auto particular", 0, 100)
    else:
        # Rama sin porcentajes → limpiar sliders + mostrar a21_2
        clear_branch("a21_1_walk", "a21_1_bike", "a21_1_pt", "a21_1_car")
        modos_percent_container.empty()

        with modos_majority_container.container():
            radio_answer(
                "a21_2",
                "¿De qué forma llega la mayoría de las personas al lugar?",
                options=[1, 2, 3, 4],
                labels_map={
                    1: "En auto particular",
                    2: "En transporte público",
                    3: "En bicicleta o similares",
                    4: "Caminando",
                },
            )

    # ---------- Conectividad ----------
    st.markdown("**Percepción de conectividad con el lugar**")
    radio_answer("a22_1", "¿Es fácil llegar al lugar?", [1, 2, 3], {1: "Sí", 2: "Más o menos", 3: "No"})
    radio_answer("a22_2", "¿Sueles llegar en automóvil particular?", [1, 2, 3], {1: "Sí", 2: "A veces", 3: "No"})
    a22_2_na = checkbox_answer("a22_2_na", "No aplica")
    if a22_2_na:
        clear_branch("a22_2")
        set_ans("a22_2", None)

    radio_answer("a22_3", "¿Has cambiado a modos más sustentables tras la intervención?", [1, 2], {1: "Sí", 2: "No"})
    a22_3_na = checkbox_answer("a22_3_na", "No aplica")
    if a22_3_na:
        clear_branch("a22_3")
        set_ans("a22_3", None)

    # ---------- Permanencia ----------
    radio_answer(
        "a23_1",
        "¿El lugar suele estar lleno?",
        [1, 2, 3, 4],
        {1: "Siempre", 2: "Frecuentemente", 3: "A veces", 4: "Casi vacío"},
    )

    # ---------- Accesibilidad del entorno ----------
    st.markdown("**Accesibilidad del entorno**")
    slider_answer("a24_1", "Accesibilidad para personas con movilidad reducida (0-100 %)", 0, 100)
    checkbox_answer("a24_1_na", "No lo sé / Sin datos")

    st.link_button("Calculadora indicadores INEGI", "https://pmm-calculadora-indicadores.streamlit.app/")

    number_answer("a24_2", "Puntaje de Accesibilidad del entorno (0-10)", 0.0, 10.0, 0.1)
    checkbox_answer("a24_2_na", "Sin dato de accesibilidad del entorno")

    # ---------- Accesibilidad dentro del lugar ----------
    st.markdown("**Accesibilidad dentro del lugar**")
    radio_answer(
        "a25_1",
        "¿Hay infraestructura para personas con movilidad reducida dentro del lugar?",
        [1, 2, 3],
        {1: "Suficiente", 2: "Insuficiente", 3: "No hay"},
    )
    number_answer("a25_2", "Puntaje de Conexión del entorno (0-10)", 0.0, 10.0, 0.1)
    checkbox_answer("a25_2_na", "Sin dato de conexión del entorno")
    radio_answer(
        "a25_3",
        "¿Personas con personas con movilidad reducida usan el lugar?",
        [1, 2, 3],
        {1: "Sí", 2: "A veces", 3: "No"},
    )

    # ---------- Primera infancia y cuidadores ----------
    st.markdown("**Accesibilidad para primera infancia y cuidadores**")
    radio_answer(
        "a26_1_p",
        "¿Hay áreas adecuadas para niñas y niños menores de 6 años y sus cuidadores?",
        [1, 2, 3],
        {1: "Sí, de calidad", 2: "Sí, pero limitadas", 3: "No existen"},
    )
    a26_2 = radio_answer("a26_2", "¿Eres cuidador/a actualmente?", [1, 2], {1: "Sí", 2: "No"})
    followup_container = st.empty()
    if a26_2 == 1:
        with followup_container.container():
            radio_answer(
                "a26_2_1",
                "Como cuidador/a, ¿qué tan satisfecho estás?",
                [1, 2, 3],
                {1: "Muy satisfecho", 2: "Aceptable", 3: "Insatisfecho"},
            )
    else:
        # Para "No" (y cualquier estado inesperado), ocultar y limpiar explícitamente.
        clear_branch("a26_2_1")
        set_ans("a26_2_1", None)
        followup_container.empty()

def pagina_A3():
    st.markdown('<div id="header_comodidad"></div>', unsafe_allow_html=True)
    st.header("Comodidad e Imagen")

    # ---------------------- SENSACIÓN DE SEGURIDAD ----------------------
    st.markdown("**Sensación de seguridad y limpieza**")
    genero = get_ans("A0_1", 2)
    if genero == 1:
        txt_seg = "¿Del 0 al 100, qué tan segura te sientes en este lugar cuando lo utilizas?"
    elif genero == 2:
        txt_seg = "¿Del 0 al 100, qué tan seguro te sientes en este lugar cuando lo utilizas?"
    else:
        txt_seg = "¿Del 0 al 100, qué tan segurx te sientes en este lugar cuando lo utilizas?"
    slider_answer("a31_1", txt_seg, 0, 100, 1)

    A0_3 = get_ans("A0_3", 2)

    # ---------------------- CUIDADO DE LA IMAGEN ----------------------
    st.markdown("**Cuidado de la imagen del lugar**")
    cuidado_resp_container = st.empty()
    cuidado_usuario_container = st.empty()
    if A0_3 == 1:
        # Rama equipo responsable → solo a32_2 visible, a32_1 se deriva
        clear_branch("a32_1")
        cuidado_usuario_container.empty()

        with cuidado_resp_container.container():
            val_resp = radio_answer(
                "a32_2",
                "Como parte del equipo responsable, ¿crees que el lugar se ha mantenido en buen estado después de la inauguración?",
                options=[1, 2, 3],
                labels_map={
                    1: "Sí, parece nuevo",
                    2: "Está en muy buen estado",
                    3: "La verdad no, está en muy malas condiciones",
                },
            )
        set_ans("a32_1", val_resp)

    else:
        # Rama usuario → solo a32_1 visible, a32_2 debe limpiarse
        clear_branch("a32_2")
        cuidado_resp_container.empty()

        with cuidado_usuario_container.container():
            radio_answer(
                "a32_1",
                "Como persona usuaria, ¿crees que el lugar se ha mantenido en buen estado después de la inauguración?",
                options=[1, 2, 3],
                labels_map={
                    1: "Sí, parece nuevo",
                    2: "Está en muy buen estado",
                    3: "La verdad no, está en muy malas condiciones",
                },
            )
        set_ans("a32_2", None)

    # ---------------------- COMODIDAD DEL LUGAR ----------------------
    st.markdown("**Comodidad del lugar**")
    radio_answer(
        "a33_1",
        "¿Crees que las áreas verdes son adecuadas y suficientes para tu comodidad a lo largo del año?",
        options=[1, 2, 3],
        labels_map={
            1: "Sí, está genial",
            2: "No está mal, pero podría mejorar",
            3: "Creo que olvidaron plantar al menos un árbol",
        },
    )
    radio_answer(
        "a33_2",
        "¿Crees que el mobiliario es lo suficientemente cómodo para usarse durante todas las estaciones del año?",
        options=[1, 2, 3],
        labels_map={
            1: "Sí, llueve, truene o relampaguee",
            2: "Hay momentos donde podría ser incómodo usarlo",
            3: "No es un lugar cómodo, ni aunque tengamos el mejor clima del mundo",
        },
    )

    # ---------------------- CAMINABILIDAD ----------------------
    st.markdown("**Caminabilidad**")
    radio_answer(
        "a34_1",
        "¿Se puede caminar cómodamente en el lugar y sus alrededores?",
        options=[1, 2, 3],
        labels_map={
            1: "Sí, totalmente",
            2: "Existen algunas barreras pero en general sí",
            3: "Es imposible hacerlo",
        },
    )

    # ---------------------- LUGARES PARA SENTARSE ----------------------
    st.markdown("**Lugares para sentarse**")
    radio_answer(
        "a35_1",
        "¿Puedes sentarte cómodamente en el lugar y sus alrededores?",
        options=[1, 2, 3],
        labels_map={
            1: "Sí",
            2: "Hay buenos asientos pero no son suficientes",
            3: "No hay infraestructura para sentarse",
        },
    )

    # ---------------------- RESILIENCIA CLIMÁTICA Y ÁREAS VERDES ----------------------
    st.markdown("**Resiliencia climática y áreas verdes**")
    resiliencia_enlace_container = st.empty()
    if A0_3 == 1:
        with resiliencia_enlace_container.container():
            slider_answer(
                "a36_1",
                "¿Qué porcentaje del lugar está diseñado con materiales locales y poco contaminantes, amigables con el medio ambiente?",
                0,
                100,
            )
            checkbox_answer("a36_1_na", "No lo sé / No hay dato sobre materiales y diseño sostenible")

            slider_answer(
                "a36_2",
                "¿Qué porcentaje del mobiliario está pensado para ser resiliente ante eventos climáticos extremos (lluvia intensa, calor extremo, tormentas, etc.)?",
                0,
                100,
            )
            checkbox_answer("a36_2_na", "No lo sé / No hay dato sobre resiliencia climática del mobiliario")

    else:
        # Si no es enlace, NO mostrar y limpiar todo
        clear_branch("a36_1", "a36_2")
        clear_branch("a36_1_na", "a36_2_na")
        resiliencia_enlace_container.empty()

        set_ans("a36_1", None)
        set_ans("a36_2", None)
        set_ans("a36_1_na", False)
        set_ans("a36_2_na", False)

    # ---------------------- PERCEPCIÓN DE AGRADO ----------------------
    st.markdown("**Percepción de agrado del lugar**")
    radio_answer(
        "a37_1",
        "¿Te parece que el lugar es agradable?",
        options=[1, 2, 3],
        labels_map={
            1: "Sí, disfruto mucho pasar tiempo aquí",
            2: "Podría mejorar, pero no está mal",
            3: "No me gusta estar aquí, trato de evitarlo lo más posible",
        },
    )


def pagina_A4():
    st.markdown('<div id="header_usos"></div>', unsafe_allow_html=True)
    st.header("Usos y actividades")

    st.markdown("**Dinamismo del lugar**")
    a41_1 = radio_answer(
        "a41_1",
        "¿Se cuenta con conteos de personas y actividades previos y posteriores a la intervención?",
        options=[1, 2],
        labels_map={1: "Sí", 2: "No"},
    )
    dinamismo_conteos_container = st.empty()
    if a41_1 == 1:
        with dinamismo_conteos_container.container():
            number_answer(
                "a41_before",
                "Personas promedio haciendo actividades diferentes ANTES de la intervención",
                0,
                1_000_000,
                1,
            )
            number_answer(
                "a41_after",
                "Personas promedio haciendo actividades diferentes DESPUÉS de la intervención",
                0,
                1_000_000,
                1,
            )
    else:
        # No conteos → limpiar inputs y usar None (mejor que 0 para no sesgar el cálculo)
        clear_branch("a41_before", "a41_after")
        set_ans("a41_before", None)
        set_ans("a41_after", None)
        dinamismo_conteos_container.empty()
    radio_answer(
        "a41_2",
        "¿Crees que la intervención ha generado un lugar más dinámico y especial para las personas que lo usan?",
        options=[1, 2, 3],
        labels_map={
            1: "¡Sí! Es muy especial para quienes lo usan habitualmente",
            2: "Ha mejorado mucho, pero tampoco siento demasiado cambio",
            3: "No he podido notar la diferencia a como era antes",
        },
    )

    st.markdown("**Que el lugar sea un referente para la comunidad**")
    radio_answer(
        "a42_1",
        "¿Crees que la comunidad considera el lugar como una referencia?",
        options=[1, 2, 3],
        labels_map={
            1: "¡Por supuesto! Se ha vuelto icónico",
            2: "Sí es más reconocido, pero ya era un referente local",
            3: "La verdad no",
        },
    )

    st.markdown("**Utilidad del lugar**")
    radio_answer(
        "a43_1",
        "¿Sientes que ha mejorado la calidad del lugar?",
        options=[1, 2, 3, 4],
        labels_map={
            1: "¡Sí! Bastante. Es mucho mejor que antes",
            2: "Sí mejoró pero no demasiado",
            3: "No noto ninguna diferencia",
            4: "Creo que empeoró la calidad",
        },
    )
    radio_answer(
        "a43_2",
        "¿Se ha vuelto más útil para la comunidad?",
        options=[1, 2, 3, 4],
        labels_map={
            1: "¡Totalmente! Es mucho más útil ahora",
            2: "Sí es más útil, pero tampoco demasiado",
            3: "La verdad tiene la misma utilidad que antes",
            4: "Creo que antes era más útil para la comunidad",
        },
    )

    st.markdown("**Actividad económica alrededor del lugar**")
    A0_3 = get_ans("A0_3", 2)
    actividad_economica_fields = st.empty()
    actividad_economica_notice = st.empty()
    if A0_3 == 1:
        actividad_economica_notice.empty()
        with actividad_economica_fields.container():
            radio_answer(
                "a44_1",
                "¿Ha aumentado el número de negocios o unidades económicas alrededor del lugar a raíz de la intervención?",
                options=[1, 2, 3],
                labels_map={1: "Sí, bastantes", 2: "Sí, por lo menos una", 3: "No, ninguna"},
            )
            a44_1_na = checkbox_answer("a44_1_na", "No aplica")
            if a44_1_na:
                clear_branch("a44_1")
                set_ans("a44_1", None)
            radio_answer(
                "a44_2",
                "¿Has percibido un mayor ingreso en tu negocio a partir de la intervención en el lugar?",
                options=[1, 2, 3],
                labels_map={1: "Sí, muy directamente", 2: "Sí, pero no sé si es por la intervención", 3: "La verdad no"},
            )
            a44_2_na = checkbox_answer("a44_2_na", "No aplica")
            if a44_2_na:
                clear_branch("a44_2")
                set_ans("a44_2", None)
    else:
        actividad_economica_fields.empty()
        with actividad_economica_notice.container():
            st.markdown("En esta evaluación no se recabó información directa de negocios o unidades económicas locales.")

        clear_branch("a44_1", "a44_2")
        clear_branch("a44_1_na", "a44_2_na")
        set_ans("a44_1", None)
        set_ans("a44_2", None)
        set_ans("a44_1_na", False)
        set_ans("a44_2_na", False)

    st.markdown("**Diversidad de actividades (regla 10+)**")
    slider_answer(
        "a45_1",
        "¿Cuántas actividades diferentes hay disponibles actualmente en el lugar?",
        0,
        10,
    )

    st.markdown("**Sentido de localidad y apropiación**")
    radio_answer(
        "a46_1",
        "¿Crees que la comunidad siente el lugar como propio?",
        options=[1, 2, 3],
        labels_map={
            1: "Sí, totalmente",
            2: "Un poco, pero no del todo",
            3: "No realmente",
        },
    )


# =========================================================
# 9. PÁGINA DE RESULTADOS
# =========================================================
def pagina_resultados():
    st.markdown('<div id="header_resultados"></div>', unsafe_allow_html=True)
    st.header("Resultados")
    ans = st.session_state["answers"]
    program_id = ans.get("program_id", "OTRO") or "OTRO"
    A0_3 = ans.get("A0_3", 2) or 2

    st.markdown("### Cargar resultados desde CSV")
    required_indicator_cols = [
        "A1.1", "A1.2", "A1.3", "A1.4", "A1.5",
        "A2.1", "A2.2", "A2.3", "A2.4", "A2.5", "A2.6",
        "A3.1", "A3.2", "A3.3", "A3.4", "A3.5", "A3.6", "A3.7",
        "A4.1", "A4.2", "A4.3", "A4.4", "A4.5", "A4.6",
    ]

    uploaded_results_csv = st.file_uploader(
        "Sube un CSV generado por esta herramienta para reconstruir las gráficas",
        type=["csv"],
        key="uploaded_results_csv",
    )

    pasted_results_text = st.text_area(
        "O pega aquí el CSV (encabezados + filas) o una sola fila en el mismo orden del CSV descargado",
        key="pasted_results_text",
        height=120,
    )

    uploaded_row = None
    uploaded_parse_error = None
    paste_parse_error = None

    pasted_row = None
    pasted_text = (pasted_results_text or "").strip()
    if pasted_text:
        try:
            if "\n" in pasted_text:
                pasted_df = pd.read_csv(StringIO(pasted_text))
                missing_cols = [c for c in required_indicator_cols if c not in pasted_df.columns]
                if pasted_df.empty:
                    paste_parse_error = "El texto pegado no contiene filas de datos."
                elif missing_cols:
                    paste_parse_error = (
                        "El texto pegado no contiene todas las columnas requeridas. "
                        f"Faltan: {', '.join(missing_cols)}"
                    )
                else:
                    if len(pasted_df) > 1:
                        pasted_row_idx = st.selectbox(
                            "Selecciona la fila pegada a visualizar",
                            options=list(range(len(pasted_df))),
                            format_func=lambda i: f"Fila pegada {i + 1}",
                            key="pasted_results_row_idx",
                        )
                    else:
                        pasted_row_idx = 0
                    pasted_row = pasted_df.iloc[int(pasted_row_idx)]
            else:
                delimiter_candidates = [",", ";", "\t", "|"]
                chosen_delimiter = max(delimiter_candidates, key=lambda d: pasted_text.count(d))
                raw_values = next(csv.reader([pasted_text], delimiter=chosen_delimiter))
                values = [v.strip() for v in raw_values]
                if len(values) != len(SHEET_COLUMNS):
                    paste_parse_error = (
                        "La fila pegada no coincide con el número esperado de columnas "
                        f"({len(values)} vs {len(SHEET_COLUMNS)})."
                    )
                else:
                    pasted_row = pd.Series({col: values[i] for i, col in enumerate(SHEET_COLUMNS)})
        except Exception as exc:
            paste_parse_error = f"No se pudo interpretar el texto pegado: {exc}"

    if uploaded_results_csv is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_results_csv)
            missing_cols = [c for c in required_indicator_cols if c not in uploaded_df.columns]
            if uploaded_df.empty:
                uploaded_parse_error = "El CSV está vacío."
            elif missing_cols:
                uploaded_parse_error = (
                    "El CSV no contiene todas las columnas requeridas. "
                    f"Faltan: {', '.join(missing_cols)}"
                )
            else:
                if len(uploaded_df) > 1:
                    selected_row_idx = st.selectbox(
                        "Selecciona la fila a visualizar",
                        options=list(range(len(uploaded_df))),
                        format_func=lambda i: f"Fila {i + 1}",
                        key="uploaded_results_csv_row_idx",
                    )
                else:
                    selected_row_idx = 0
                uploaded_row = uploaded_df.iloc[int(selected_row_idx)]
        except Exception as exc:
            uploaded_parse_error = f"No se pudo leer el CSV: {exc}"

    if pasted_row is not None:
        uploaded_row = pasted_row
        st.info("Usando datos pegados para reconstruir resultados.")

    if paste_parse_error:
        st.warning(paste_parse_error)

    if uploaded_parse_error:
        st.warning(uploaded_parse_error)

    def _min_visible_score(value, minimum=20.0, include_none=False):
        """Asegura que los scores tengan un valor mínimo visible."""
        if value is None:
            return minimum if include_none else None
        try:
            num = float(value)
        except (TypeError, ValueError):
            return value
        return max(minimum, num)

    def _apply_min_visible_scores(scores_dict, minimum=20.0, include_none=False):
        return {
            key: _min_visible_score(val, minimum, include_none=include_none)
            for key, val in scores_dict.items()
        }

    # ========== A1 ==========
    a11_1_val = ans.get("a11_1")
    a11_1 = None if ans.get("a11_1_na") else (float(a11_1_val) if a11_1_val is not None else None)
    a11_2_val = ans.get("a11_2")
    if ans.get("a11_2_na"):
        a11_2 = None
    elif a11_2_val is None:
        a11_2 = None
    else:
        a11_2_num = float(a11_2_val)
        a11_2 = a11_2_num * 10 if a11_2_num <= 10 else a11_2_num

    a12_1 = ans.get("a12_1")
    a12_2_1 = ans.get("a12_2_1")
    a12_2_2 = ans.get("a12_2_2")
    a12_3 = ans.get("a12_3")
    a13_1 = ans.get("a13_1")
    a14_1 = ans.get("a14_1")
    a15_1 = ans.get("a15_1")
    a15_2 = ans.get("a15_2")

    A1_1 = calc_A1_1(a11_1, a11_2)
    A1_2 = calc_A1_2(a12_1, a12_2_1, a12_2_2, a12_3)
    A1_3 = calc_A1_3(a12_1, a13_1, a12_2_1, a12_2_2)
    A1_4 = calc_A1_4(A1_3, a14_1)
    A1_5 = calc_A1_5(a12_3, a15_1, a15_2)

    A1_indicators_raw = {
        "A1.1": A1_1,
        "A1.2": A1_2,
        "A1.3": A1_3,
        "A1.4": A1_4,
        "A1.5": A1_5,
    }
    A1_indicators = impute_missing_with_penalty(A1_indicators_raw, penalty=40.0)

    # ========== A3 ==========
    a31_raw = ans.get("a31_1")
    a31_1 = float(a31_raw) if a31_raw is not None else None
    a32_1 = ans.get("a32_1")
    a32_2 = ans.get("a32_2")
    a33_1 = ans.get("a33_1")
    a33_2 = ans.get("a33_2")
    a34_1 = ans.get("a34_1")
    a35_1 = ans.get("a35_1")

    a36_1_val = ans.get("a36_1")
    a36_1 = None if ans.get("a36_1_na") else (float(a36_1_val) if a36_1_val is not None else None)
    a36_2_val = ans.get("a36_2")
    a36_2 = None if ans.get("a36_2_na") else (float(a36_2_val) if a36_2_val is not None else None)

    a37_1 = ans.get("a37_1")

    A3_1 = calc_A3_1(a31_1, a14_1)
    A3_2 = calc_A3_2(a32_1, a32_2, a14_1, A0_3)
    A3_3 = calc_A3_3(a33_1, a33_2)
    A3_4 = calc_A3_4(a34_1)
    A3_5 = calc_A3_5(a35_1)
    A3_6 = calc_A3_6(a36_1, a36_2)
    A3_7 = calc_A3_7(a37_1)

    A3_indicators_raw = {
        "A3.1": A3_1,
        "A3.2": A3_2,
        "A3.3": A3_3,
        "A3.4": A3_4,
        "A3.5": A3_5,
        "A3.6": A3_6,
        "A3.7": A3_7,
    }
    A3_indicators = impute_missing_with_penalty(A3_indicators_raw, penalty=40.0)

    # ========== A2 ==========
    a21_use_percent = ans.get("a21_use_percent")
    if a21_use_percent == 1:
        a21_1_walk = float(ans.get("a21_1_walk", 0))
        a21_1_bike = float(ans.get("a21_1_bike", 0))
        a21_1_pt = float(ans.get("a21_1_pt", 0))
        a21_1_car = float(ans.get("a21_1_car", 0))
        a21_2 = None
    else:
        a21_1_walk = a21_1_bike = a21_1_pt = a21_1_car = None
        a21_2 = ans.get("a21_2")

    a22_1 = ans.get("a22_1")
    a22_2 = None if ans.get("a22_2_na") else ans.get("a22_2")
    a22_3 = None if ans.get("a22_3_na") else ans.get("a22_3")
    a23_1 = ans.get("a23_1")

    a24_1_val = ans.get("a24_1")
    a24_1 = None if ans.get("a24_1_na") else (float(a24_1_val) if a24_1_val is not None else None)
    a24_2_val = ans.get("a24_2")
    a24_2 = None if ans.get("a24_2_na") else (float(a24_2_val) * 10 if a24_2_val is not None else None)

    a25_2_val = ans.get("a25_2")
    a25_2 = None if ans.get("a25_2_na") else (float(a25_2_val) * 10 if a25_2_val is not None else None)

    a25_1 = ans.get("a25_1")
    a25_3 = ans.get("a25_3")
    a26_1_p = ans.get("a26_1_p")
    a26_2 = ans.get("a26_2")
    a26_2_1 = ans.get("a26_2_1")

    A2_1 = calc_A2_1(a21_1_walk, a21_1_bike, a21_1_pt, a21_1_car, a21_2)
    A2_2 = calc_A2_2(a22_1, a22_2, a22_3)
    A2_3 = calc_A2_3(a23_1)
    A2_4 = calc_A2_4(a24_1, a24_2)
    A2_5 = calc_A2_5(a25_1, a25_2, a25_3)
    A2_6 = calc_A2_6(a26_1_p, a26_2, a26_2_1)

    A2_indicators_raw = {
        "A2.1": A2_1,
        "A2.2": A2_2,
        "A2.3": A2_3,
        "A2.4": A2_4,
        "A2.5": A2_5,
        "A2.6": A2_6,
    }
    A2_indicators = impute_missing_with_penalty(A2_indicators_raw, penalty=40.0)

    # ========== A4 ==========
    a41_1 = ans.get("a41_1")
    a41_before = ans.get("a41_before", 0)
    a41_after = ans.get("a41_after", 0)
    a41_2 = ans.get("a41_2")
    a42_1 = ans.get("a42_1")
    a43_1 = ans.get("a43_1")
    a43_2 = ans.get("a43_2")
    a44_1 = None if ans.get("a44_1_na") else ans.get("a44_1")
    a44_2 = None if ans.get("a44_2_na") else ans.get("a44_2")
    a45_1 = ans.get("a45_1", 0)
    a46_1 = ans.get("a46_1")

    A4_1 = calc_A4_1(a41_1, a41_before, a41_after, a41_2)
    A4_2 = calc_A4_2(a42_1)
    A4_3 = calc_A4_3(a43_1, a43_2)
    A4_5 = calc_A4_5(a45_1)
    S461 = score_1_3_100_50_10(a46_1)
    S462 = score_A1_5_1(a15_1)
    A4_6 = calc_A4_6(S461, S462, A1_1)
    A4_4 = calc_A4_4(A0_3, a44_1, a44_2, A4_1, A4_2, A4_3, A4_5, A4_6)

    A4_indicators_raw = {
        "A4.1": A4_1,
        "A4.2": A4_2,
        "A4.3": A4_3,
        "A4.4": A4_4,
        "A4.5": A4_5,
        "A4.6": A4_6,
    }
    A4_indicators = impute_missing_with_penalty(A4_indicators_raw, penalty=40.0)

    # ===== Intangibles =====
    intangibles_A1_raw = calc_intangibles_A1(A1_1, A1_2, A1_3, A1_4, A1_5)
    intangibles_A2_raw = calc_intangibles_A2(A2_1, A2_2, A2_3, A2_4, A2_5, A2_6, A3_4)
    intangibles_A3_raw = calc_intangibles_A3(A3_1, A3_2, A3_3, A3_4, A3_5, A3_6, A3_7)
    intangibles_A4_raw = calc_intangibles_A4(
        A4_1,
        A4_2,
        A4_3,
        A4_4,
        A4_5,
        A4_6,
        A1_2,
        A1_3,
        A3_6,
        intangibles_A3_raw.get("Agradable"),
    )

    def zeros_to_none(d):
        return {k: (None if v == 0 else v) for k, v in d.items()}

    intangibles_A1_with_none = zeros_to_none(intangibles_A1_raw)
    intangibles_A2_with_none = zeros_to_none(intangibles_A2_raw)
    intangibles_A3_with_none = zeros_to_none(intangibles_A3_raw)
    intangibles_A4_with_none = zeros_to_none(intangibles_A4_raw)

    intangibles_A1 = impute_missing_with_penalty(intangibles_A1_with_none, penalty=40.0)
    intangibles_A2 = impute_missing_with_penalty(intangibles_A2_with_none, penalty=40.0)
    intangibles_A3 = impute_missing_with_penalty(intangibles_A3_with_none, penalty=40.0)
    intangibles_A4 = impute_missing_with_penalty(intangibles_A4_with_none, penalty=40.0)

    # ===== Totales y global =====
    A1_total = compute_attribute_total(program_id, "A1", A1_indicators)
    A2_total = compute_attribute_total(program_id, "A2", A2_indicators)
    A3_total = compute_attribute_total(program_id, "A3", A3_indicators)
    A4_total = compute_attribute_total(program_id, "A4", A4_indicators)

    section_scores = compute_section_scores(A1_total, A2_total, A3_total, A4_total)
    global_score = compute_global_score(
        program_id,
        section_scores,
        sostenible_score=intangibles_A4.get("Sostenible"),
        sostenible_weight=0.15,
        conmemorativo_score=intangibles_A4.get("Conmemorativo"),
        conmemorativo_weight=0.10,
    )

    if uploaded_row is not None:
        try:
            def _row_float_or_none(col_name):
                if col_name not in uploaded_row.index:
                    return None
                val = uploaded_row[col_name]
                if pd.isna(val):
                    return None
                parsed = to_float_or_none(val)
                if parsed is None:
                    return None
                if isinstance(parsed, float) and not np.isfinite(parsed):
                    return None
                return parsed

            uploaded_program_id = str(uploaded_row.get("programa", program_id) or program_id)

            A1_indicators = {
                "A1.1": _row_float_or_none("A1.1"),
                "A1.2": _row_float_or_none("A1.2"),
                "A1.3": _row_float_or_none("A1.3"),
                "A1.4": _row_float_or_none("A1.4"),
                "A1.5": _row_float_or_none("A1.5"),
            }
            A2_indicators = {
                "A2.1": _row_float_or_none("A2.1"),
                "A2.2": _row_float_or_none("A2.2"),
                "A2.3": _row_float_or_none("A2.3"),
                "A2.4": _row_float_or_none("A2.4"),
                "A2.5": _row_float_or_none("A2.5"),
                "A2.6": _row_float_or_none("A2.6"),
            }
            A3_indicators = {
                "A3.1": _row_float_or_none("A3.1"),
                "A3.2": _row_float_or_none("A3.2"),
                "A3.3": _row_float_or_none("A3.3"),
                "A3.4": _row_float_or_none("A3.4"),
                "A3.5": _row_float_or_none("A3.5"),
                "A3.6": _row_float_or_none("A3.6"),
                "A3.7": _row_float_or_none("A3.7"),
            }
            A4_indicators = {
                "A4.1": _row_float_or_none("A4.1"),
                "A4.2": _row_float_or_none("A4.2"),
                "A4.3": _row_float_or_none("A4.3"),
                "A4.4": _row_float_or_none("A4.4"),
                "A4.5": _row_float_or_none("A4.5"),
                "A4.6": _row_float_or_none("A4.6"),
            }

            A1_1_u, A1_2_u, A1_3_u, A1_4_u, A1_5_u = (
                A1_indicators["A1.1"],
                A1_indicators["A1.2"],
                A1_indicators["A1.3"],
                A1_indicators["A1.4"],
                A1_indicators["A1.5"],
            )
            A2_1_u, A2_2_u, A2_3_u, A2_4_u, A2_5_u, A2_6_u = (
                A2_indicators["A2.1"],
                A2_indicators["A2.2"],
                A2_indicators["A2.3"],
                A2_indicators["A2.4"],
                A2_indicators["A2.5"],
                A2_indicators["A2.6"],
            )
            A3_1_u, A3_2_u, A3_3_u, A3_4_u, A3_5_u, A3_6_u, A3_7_u = (
                A3_indicators["A3.1"],
                A3_indicators["A3.2"],
                A3_indicators["A3.3"],
                A3_indicators["A3.4"],
                A3_indicators["A3.5"],
                A3_indicators["A3.6"],
                A3_indicators["A3.7"],
            )
            A4_1_u, A4_2_u, A4_3_u, A4_4_u, A4_5_u, A4_6_u = (
                A4_indicators["A4.1"],
                A4_indicators["A4.2"],
                A4_indicators["A4.3"],
                A4_indicators["A4.4"],
                A4_indicators["A4.5"],
                A4_indicators["A4.6"],
            )

            intangibles_A1_raw = calc_intangibles_A1(A1_1_u, A1_2_u, A1_3_u, A1_4_u, A1_5_u)
            intangibles_A2_raw = calc_intangibles_A2(A2_1_u, A2_2_u, A2_3_u, A2_4_u, A2_5_u, A2_6_u, A3_4_u)
            intangibles_A3_raw = calc_intangibles_A3(A3_1_u, A3_2_u, A3_3_u, A3_4_u, A3_5_u, A3_6_u, A3_7_u)
            intangibles_A4_raw = calc_intangibles_A4(
                A4_1_u,
                A4_2_u,
                A4_3_u,
                A4_4_u,
                A4_5_u,
                A4_6_u,
                A1_2_u,
                A1_3_u,
                A3_6_u,
                intangibles_A3_raw.get("Agradable"),
            )

            intangibles_A1 = impute_missing_with_penalty({k: (None if v == 0 else v) for k, v in intangibles_A1_raw.items()}, penalty=40.0)
            intangibles_A2 = impute_missing_with_penalty({k: (None if v == 0 else v) for k, v in intangibles_A2_raw.items()}, penalty=40.0)
            intangibles_A3 = impute_missing_with_penalty({k: (None if v == 0 else v) for k, v in intangibles_A3_raw.items()}, penalty=40.0)
            intangibles_A4 = impute_missing_with_penalty({k: (None if v == 0 else v) for k, v in intangibles_A4_raw.items()}, penalty=40.0)

            A1_total = _row_float_or_none("A1_total")
            A2_total = _row_float_or_none("A2_total")
            A3_total = _row_float_or_none("A3_total")
            A4_total = _row_float_or_none("A4_total")
            if A1_total is None:
                A1_total = compute_attribute_total(uploaded_program_id, "A1", A1_indicators)
            if A2_total is None:
                A2_total = compute_attribute_total(uploaded_program_id, "A2", A2_indicators)
            if A3_total is None:
                A3_total = compute_attribute_total(uploaded_program_id, "A3", A3_indicators)
            if A4_total is None:
                A4_total = compute_attribute_total(uploaded_program_id, "A4", A4_indicators)

            global_score = _row_float_or_none("global_score")
            if global_score is None:
                section_scores = compute_section_scores(A1_total, A2_total, A3_total, A4_total)
                global_score = compute_global_score(
                    uploaded_program_id,
                    section_scores,
                    sostenible_score=intangibles_A4.get("Sostenible"),
                    sostenible_weight=0.15,
                    conmemorativo_score=intangibles_A4.get("Conmemorativo"),
                    conmemorativo_weight=0.10,
                )

            program_id = uploaded_program_id
            nombre_lugar = str(uploaded_row.get("nombre_lugar", ans.get("nombre_lugar", "")) or "").strip()
            nombre_eval = str(uploaded_row.get("nombre_evaluador", ans.get("nombre_evaluador", "")) or "").strip()
            st.info("Mostrando gráficas y resultados desde el CSV cargado.")
        except Exception as exc:
            st.warning(f"No se pudo reconstruir resultados desde el CSV cargado. Se usan los datos actuales del formulario. Detalle: {exc}")
            uploaded_row = None

    # ===== Resumen numérico =====
    st.subheader("Resumen de atributos")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Encuentro", f"{A1_total:.1f}")
    c2.metric("Conexiones y Accesos", f"{A2_total:.1f}")
    c3.metric("Comodidad e Imagen", f"{A3_total:.1f}")
    c4.metric("Usos y Actividades", f"{A4_total:.1f}")
    c5.metric("Resultado global", f"{global_score:.1f}")

    if uploaded_row is None:
        nombre_lugar = ans.get("nombre_lugar", "").strip()
        nombre_eval = ans.get("nombre_evaluador", "").strip()
    st.markdown(
        f"**Lugar:** {nombre_lugar or 'Sin nombre'} — "
        f"**Evaluado por:** {nombre_eval or 'Sin especificar'}"
    )

    # ===== Configuración de sectores y scores para las gráficas =====
    sector_names = [
        "Usos y Actividades",
        "Comodidad e Imagen",
        "Conexiones y Accesos",
        "Encuentro",
    ]
    SECTOR_CONFIG = {
        "Encuentro": {
            "middle_labels": [
                "Diversidad",
                "Cuidado",
                "Comunidad",
                "Compartido",
                "Símbolos",
                "Orgullo",
                "Amigable",
                "Interactivo",
            ],
            "outer_labels": ["A1.1", "A1.2", "A1.3", "A1.4", "A1.5"],
        },
        "Conexiones y Accesos": {
            "middle_labels": [
                "Cercano",
                "Conectado",
                "Conveniente",
                "Accesible\n(movilidad reducida)",
                "Accesible (primera\ninfancia y cuidadores)",
                "Transitable",
            ],
            "outer_labels": ["A2.1", "A2.2", "A2.3", "A2.4", "A2.5", "A2.6"],
        },
        "Comodidad e Imagen": {
            "middle_labels": [
                "Limpio",
                "Seguro",
                "Sentable",
                "Agradable",
                "Verde",
                "Caminable",
                "Resiliencia climática",
            ],
            "outer_labels": [
                "A3.1",
                "A3.2",
                "A3.3",
                "A3.4",
                "A3.5",
                "A3.6",
                "A3.7",
            ],
        },
        "Usos y Actividades": {
            "middle_labels": [
                "Dinámico",
                "Especial",
                "Real",
                "Útil",
                "Local",
                "Sostenible",
                "Conmemorativo",
                "Pertenencia",
            ],
            "outer_labels": ["A4.1", "A4.2", "A4.3", "A4.4", "A4.5", "A4.6"],
        },
    }

    scores_middle = {
        "Usos y Actividades": _apply_min_visible_scores(intangibles_A4, include_none=True),
        "Comodidad e Imagen": _apply_min_visible_scores(intangibles_A3, include_none=True),
        "Conexiones y Accesos": _apply_min_visible_scores(intangibles_A2, include_none=True),
        "Encuentro": _apply_min_visible_scores(intangibles_A1, include_none=True),
    }
    scores_outer = {
        "Usos y Actividades": _apply_min_visible_scores(A4_indicators, include_none=True),
        "Comodidad e Imagen": _apply_min_visible_scores(A3_indicators, include_none=True),
        "Conexiones y Accesos": _apply_min_visible_scores(A2_indicators, include_none=True),
        "Encuentro": _apply_min_visible_scores(A1_indicators, include_none=True),
    }

    if not MATPLOTLIB_AVAILABLE:
        st.error("No se pudo cargar matplotlib para generar las gráficas de resultados.")
        st.info(f"Instala el paquete en el mismo entorno que ejecuta Streamlit: `{sys.executable} -m pip install matplotlib`")
        if MATPLOTLIB_ERROR:
            st.caption(f"Detalle técnico: {MATPLOTLIB_ERROR}")
        return

    if not PYCIRCLIZE_AVAILABLE:
        st.error("No se pudo cargar pycirclize para generar las gráficas de resultados.")
        st.info(f"Instala el paquete en el mismo entorno que ejecuta Streamlit: `{sys.executable} -m pip install pycirclize`")
        if PYCIRCLIZE_ERROR:
            st.caption(f"Detalle técnico: {PYCIRCLIZE_ERROR}")
        return

    # ──────────────────────────────────────────
    # Rueda global (UI sin textos del gráfico)
    # ──────────────────────────────────────────
    from datetime import datetime

    # Toggle de metadata
    st.markdown("**Metadata**")
    show_metadata = st.checkbox(
        "Mostrar metadata",
        value=True,
        key="show_metadata_wheel",
    )

    st.markdown("---")

    # Fuente local (opcional)
    for font_file in [
        BASE_DIR / "Poppins" / "Poppins-Regular.ttf",
        BASE_DIR / "Poppins" / "Poppins-SemiBold.ttf",
    ]:
        if font_file.exists():
            font_manager.fontManager.addfont(str(font_file))
    plt.rcParams["font.family"] = "Poppins"

    # Colores (look)
    track_colors = {
        "Usos y Actividades": {"inner": "#E45B48", "middle": "#E79A90", "outer": "#EBCBC2"},
        "Comodidad e Imagen": {"inner": "#A5BB39", "middle": "#BFCA73", "outer": "#DBE0B2"},
        "Conexiones y Accesos": {"inner": "#275A9F", "middle": "#7297C2", "outer": "#B4C9DE"},
        "Encuentro": {"inner": "#575A99", "middle": "#9EA2C6", "outer": "#BEC1CC"},
    }

    # Radios
    r_center_max = 13
    r_inner_min, r_inner_max = 15, 40
    r_mid_min, r_mid_max = 42, 73
    r_outer_min, r_outer_max = 75, 108

    # Sectores (quita la inclinación por espacios entre cuadrantes)
    sectors = {name: 90 for name in sector_names}
    circos = Circos(sectors, start=0, end=360, space=0)

    # Scores (igual funcionalidad)
    sector_scores = {}
    for name in sector_names:
        mid_labels = SECTOR_CONFIG[name]["middle_labels"]
        out_labels = SECTOR_CONFIG[name]["outer_labels"]
        middle_pcts = [scores_middle.get(name, {}).get(lbl, 0.0) for lbl in mid_labels]
        outer_pcts = [scores_outer.get(name, {}).get(lbl, 0.0) for lbl in out_labels]
        sector_scores[name] = {
            "middle_pcts": middle_pcts,
            "outer_pcts": outer_pcts,
        }

    # Dibujo de anillos (mantiene lógica de resultados)
    for sector in circos.sectors:
        name = sector.name
        colors = track_colors[name]
        length = sector.size

        middle_pcts = sector_scores[name]["middle_pcts"]
        outer_pcts = sector_scores[name]["outer_pcts"]

        # INNER
        t_inner = sector.add_track((r_inner_min, r_inner_max))
        t_inner.rect(0, length, fc=colors["inner"], ec="white", lw=4)

        # MIDDLE
        if middle_pcts:
            blk = length / len(middle_pcts)
            for i, pct in enumerate(middle_pcts):
                start, end = i * blk, (i + 1) * blk
                ratio = max(0.0, min(1.0, nz(pct) / 100.0))
                if ratio > 0:
                    t = sector.add_track((r_mid_min, r_mid_min + (r_mid_max - r_mid_min) * ratio))
                    t.rect(start, end, fc=colors["middle"], ec="white", lw=4)

        # OUTER
        if outer_pcts:
            blk = length / len(outer_pcts)
            for i, pct in enumerate(outer_pcts):
                start, end = i * blk, (i + 1) * blk
                ratio = max(0.0, min(1.0, nz(pct) / 100.0))
                if ratio > 0:
                    t = sector.add_track((r_outer_min, r_outer_min + (r_outer_max - r_outer_min) * ratio))
                    t.rect(start, end, fc=colors["outer"], ec="white", lw=4)
    # Figura final
    fig_size = (9.6, 10.2) if show_metadata else (9.6, 9.6)
    fig = circos.plotfig(figsize=fig_size, dpi=110)
    ax = fig.axes[0]

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.patch.set_facecolor("#FFFFFF")
    ax.patch.set_facecolor("#FFFFFF")

    # Metadata (mantén la tuya, solo ajusta posiciones)
    if show_metadata:
        # Título (solo cuando metadata está activa)
        place_meta = " · ".join(
            x for x in [nombre_lugar, ans.get("a0_municipio"), ans.get("a0_estado")] if x
        )
        fig.text(
            0.5, 1.05, place_meta or "Lugar",
            ha="center", va="center",
            fontsize=24, color="#232544", fontweight="bold"
        )
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        fig.text(0.14, 0.035, timestamp, ha="center", va="center", fontsize=10, color="#777777", style="italic")
        fig.text(0.86, 0.035, nombre_eval or "Evaluador", ha="center", va="center", fontsize=10, color="#777777")
        fig.subplots_adjust(top=0.835, bottom=0.07, left=0.02, right=0.98)
    else:
        fig.subplots_adjust(top=0.835, bottom=0.04, left=0.02, right=0.98)

    st.pyplot(fig, use_container_width=True)

    # ===== Gráfica de fortalezas y oportunidades =====
    attribute_scores = {
        "Encuentro": A1_total,
        "Conexiones y Accesos": A2_total,
        "Comodidad e Imagen": A3_total,
        "Usos y Actividades": A4_total,
    }

    ordered_attributes = sorted(attribute_scores.keys(), key=lambda name: attribute_scores[name], reverse=True)
    bar_labels = ordered_attributes
    bar_values = [attribute_scores[name] for name in ordered_attributes]
    bar_colors = [track_colors[name]["inner"] for name in ordered_attributes]

    fig_attr, ax_attr = plt.subplots(figsize=(8.6, 4.0), dpi=110)
    x_pos = np.arange(len(bar_labels))
    bars = ax_attr.bar(x_pos, bar_values, color=bar_colors, width=0.62)

    ax_attr.set_ylim(0, 112)
    ax_attr.set_xticks(x_pos)
    ax_attr.set_xticklabels(bar_labels, fontsize=10, fontname="Poppins", color="#232544", rotation=0, ha="center")
    ax_attr.set_title(
        "Comparación",
        fontsize=15,
        fontname="Poppins",
        color="#232544",
        fontweight="bold",
        pad=42,
    )
    chart_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    place_meta_chart = " · ".join(
        x for x in [nombre_lugar, ans.get("a0_municipio"), ans.get("a0_estado")] if x
    )
    meta_text = f"Lugar: {place_meta_chart or 'Sin nombre'}   |   Evaluador: {nombre_eval or 'Sin especificar'}   |   {chart_timestamp}"
    ax_attr.text(
        0.5,
        1.08,
        meta_text,
        transform=ax_attr.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        fontname="Poppins",
        color="#777777",
    )

    ax_attr.grid(False)
    ax_attr.set_yticks([])

    for spine in ["top", "right", "left", "bottom"]:
        ax_attr.spines[spine].set_visible(False)

    for bar, value in zip(bars, bar_values):
        ax_attr.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 2.0, 108.0),
            f"{value:.1f}",
            va="bottom",
            ha="center",
            fontsize=10,
            fontname="Poppins",
            color="#232544",
            fontweight="bold",
        )

    fig_attr.patch.set_facecolor("#FFFFFF")
    ax_attr.patch.set_facecolor("#FFFFFF")
    fig_attr.subplots_adjust(top=0.76, bottom=0.2)
    st.pyplot(fig_attr, use_container_width=True)

    # ===== Gráfica de puntos y tabla: intangibles (mejores/peores/atención) =====
    all_intangibles = []
    for atributo, intangibles_dict in [
        ("Encuentro", intangibles_A1),
        ("Conexiones y Accesos", intangibles_A2),
        ("Comodidad e Imagen", intangibles_A3),
        ("Usos y Actividades", intangibles_A4),
    ]:
        for intangible_name, intangible_value in intangibles_dict.items():
            if intangible_value is None:
                continue
            all_intangibles.append(
                {
                    "Atributo": atributo,
                    "Intangible": intangible_name,
                    "Valor": float(intangible_value),
                }
            )

    if all_intangibles:
        df_all_intangibles = pd.DataFrame(all_intangibles).sort_values("Valor", ascending=False).reset_index(drop=True)

        top_n = min(5, len(df_all_intangibles))
        bottom_n = min(5, len(df_all_intangibles))
        top_indices = set(df_all_intangibles.head(top_n).index.tolist())
        bottom_indices = set(df_all_intangibles.tail(bottom_n).index.tolist())

        median_value = float(df_all_intangibles["Valor"].median())
        candidate_median = df_all_intangibles.copy()
        candidate_median["dist_median"] = (candidate_median["Valor"] - median_value).abs()
        candidate_median = candidate_median[~candidate_median.index.isin(top_indices | bottom_indices)]
        median_n = min(5, len(candidate_median))
        median_indices = set(candidate_median.nsmallest(median_n, "dist_median").index.tolist())

        selected_indices = list(top_indices | bottom_indices | median_indices)
        df_selected = df_all_intangibles.loc[selected_indices].copy()

        def classify_intangible(row_index: int) -> str:
            if row_index in top_indices:
                return "Mejores"
            if row_index in bottom_indices:
                return "Peores"
            return "Mediana"

        df_selected["Categoría"] = [classify_intangible(i) for i in df_selected.index.tolist()]

        attribute_colors = {
            "Encuentro": track_colors["Encuentro"]["inner"],
            "Conexiones y Accesos": track_colors["Conexiones y Accesos"]["inner"],
            "Comodidad e Imagen": track_colors["Comodidad e Imagen"]["inner"],
            "Usos y Actividades": track_colors["Usos y Actividades"]["inner"],
        }

        semaforo_bg = {
            "Peores": "#FDECEC",
            "Media": "#FFF4E5",
            "Mejores": "#EAF7EC",
        }
        semaforo_title = {
            "Peores": "#C62828",
            "Media": "#EF6C00",
            "Mejores": "#2E7D32",
        }

        df_selected["Categoría"] = df_selected["Categoría"].replace({"Mediana": "Media"})
        panel_order = ["Peores", "Media", "Mejores"]
        panel_display_names = {
            "Peores": "Debilidades",
            "Media": "Promedio",
            "Mejores": "Fortalezas",
        }
        panel_data = {
            category: df_selected[df_selected["Categoría"] == category].sort_values("Valor", ascending=False).reset_index(drop=True)
            for category in panel_order
        }

        max_rows = max((len(panel_data[c]) for c in panel_order), default=1)
        fig_height = max(4.8, 2.8 + max_rows * 0.65)
        fig_int, ax_int = plt.subplots(figsize=(13.0, fig_height), dpi=110)
        ax_int.axis("off")

        col_w = 0.305
        col_gap = 0.0225
        col_x = [0.0, col_w + col_gap, 2 * (col_w + col_gap)]

        for x0, category in zip(col_x, panel_order):
            ax_int.add_patch(
                plt.Rectangle(
                    (x0, 0.05),
                    col_w,
                    0.88,
                    transform=ax_int.transAxes,
                    facecolor=semaforo_bg[category],
                    edgecolor="none",
                )
            )
            ax_int.text(
                x0 + (col_w / 2),
                0.89,
                panel_display_names[category],
                transform=ax_int.transAxes,
                ha="center",
                va="center",
                fontsize=18,
                fontname="Poppins",
                color=semaforo_title[category],
                fontweight="bold",
            )

            df_cat = panel_data[category]
            y_start = 0.80
            row_gap = 0.13
            for row_idx, (_, row) in enumerate(df_cat.iterrows()):
                y_text = y_start - row_idx * row_gap
                score_0_10 = float(row["Valor"]) / 10.0
                ax_int.text(
                    x0 + 0.02,
                    y_text,
                    f"• {score_0_10:.1f}  {row['Intangible']}",
                    transform=ax_int.transAxes,
                    ha="left",
                    va="center",
                    fontsize=13,
                    fontname="Poppins",
                    color=attribute_colors.get(row["Atributo"], "#232544"),
                    fontweight="semibold",
                )

        ax_int.text(
            0.5,
            0.98,
            "Cualidades Intangibles",
            transform=ax_int.transAxes,
            ha="center",
            va="center",
            fontsize=20,
            fontname="Poppins",
            color="#232544",
            fontweight="bold",
        )

        # Leyenda por atributo (color del texto)
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", label=attr, markerfacecolor=col, markersize=8)
            for attr, col in attribute_colors.items()
        ]
        fig_int.legend(
            handles=legend_handles,
            title="Color por atributo",
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=8.5,
            title_fontsize=9,
            bbox_to_anchor=(0.5, 0.01),
        )

        fig_int.patch.set_facecolor("#FFFFFF")
        fig_int.subplots_adjust(left=0.03, right=0.99, top=0.93, bottom=0.06)
        st.pyplot(fig_int, use_container_width=True)

        peores_nombres = ", ".join(panel_data["Peores"]["Intangible"].tolist())
        media_nombres = ", ".join(panel_data["Media"]["Intangible"].tolist())
        mejores_nombres = ", ".join(panel_data["Mejores"]["Intangible"].tolist())

        st.markdown("#### Interpretación y recomendaciones")
        st.write(
            "Los resultados muestran tres grupos claros de intangibles. "
            f"En **Fortalezas** destacan: {mejores_nombres}. "
            f"En **Promedio** se ubican: {media_nombres}. "
            f"En **Debilidades** aparecen: {peores_nombres}."
        )
        st.write(
            "**Recomendaciones:** (1) proteger y replicar en otros atributos las prácticas que sostienen los intangibles de Mejores; "
            "(2) para los de Media, definir acciones de corto plazo con metas de mejora por trimestre para moverlos al grupo alto; "
            "(3) en Peores, priorizar intervenciones focalizadas y seguimiento frecuente, empezando por los factores de uso, acceso, "
            "cuidado y organización comunitaria que estén limitando su desempeño."
        )

    attribute_colors = {
        "Encuentro": track_colors["Encuentro"]["inner"],
        "Conexiones y Accesos": track_colors["Conexiones y Accesos"]["inner"],
        "Comodidad e Imagen": track_colors["Comodidad e Imagen"]["inner"],
        "Usos y Actividades": track_colors["Usos y Actividades"]["inner"],
    }
    df_indicator_filtered = None
    df_intang_filtered = None

    st.markdown("---")
    st.markdown("### Indicadores por atributo")
    if not PLOTLY_AVAILABLE:
        detail = f" Detalle técnico: {PLOTLY_ERROR}" if PLOTLY_ERROR else ""
        st.warning(f"No se pudo cargar Plotly para visualizaciones interactivas.{detail}")
    else:
        indicator_labels_by_sector = {
            "Encuentro": {
                "A1.1": "Diversidad demográfica",
                "A1.2": "Redes ciudadanas",
                "A1.3": "Voluntariado",
                "A1.4": "Cuidado del lugar",
                "A1.5": "Uso nocturno",
            },
            "Conexiones y Accesos": {
                "A2.1": "Modos de transporte",
                "A2.2": "Conectividad con el lugar",
                "A2.3": "Permanencia",
                "A2.4": "Accesibilidad del entorno",
                "A2.5": "Accesibilidad dentro del lugar (movilidad reducida)",
                "A2.6": "Accesibilidad dentro del lugar (primera infancia y cuidadores)",
            },
            "Comodidad e Imagen": {
                "A3.1": "Seguridad y limpieza",
                "A3.2": "Cuidado de la imagen",
                "A3.3": "Comodidad",
                "A3.4": "Caminabilidad",
                "A3.5": "Lugares para sentarse",
                "A3.6": "Resiliencia climática y áreas verdes",
                "A3.7": "Ser agradable",
            },
            "Usos y Actividades": {
                "A4.1": "Dinamismo",
                "A4.2": "Referente",
                "A4.3": "Utilidad",
                "A4.4": "Actividad económica",
                "A4.5": "Diversidad de actividades",
                "A4.6": "Localidad",
            },
        }

        sector_to_indicators_data = {
            "Encuentro": A1_indicators,
            "Conexiones y Accesos": A2_indicators,
            "Comodidad e Imagen": A3_indicators,
            "Usos y Actividades": A4_indicators,
        }

        indicator_rows = []
        global_order_idx = 1
        for sector_name in ["Encuentro", "Conexiones y Accesos", "Comodidad e Imagen", "Usos y Actividades"]:
            labels_map = indicator_labels_by_sector[sector_name]
            indicators_map = sector_to_indicators_data[sector_name]
            local_pos = 1
            for code, label in labels_map.items():
                value = indicators_map.get(code)
                if value is None:
                    local_pos += 1
                    continue
                indicator_rows.append(
                    {
                        "Atributo": sector_name,
                        "Indicador": label,
                        "Valor": float(value),
                        "OrdenGlobal": global_order_idx,
                        "PosicionAtributo": local_pos,
                    }
                )
                global_order_idx += 1
                local_pos += 1

        if indicator_rows:
            df_indicator_plot = pd.DataFrame(indicator_rows)
            all_attrs = ["Encuentro", "Conexiones y Accesos", "Comodidad e Imagen", "Usos y Actividades"]
            all_indicator_names = df_indicator_plot["Indicador"].tolist()

            st.markdown("**Filtrar atributos (indicadores)**")
            attr_cols_ind = st.columns(2)
            selected_attrs_ind = []
            for idx, attr_name in enumerate(all_attrs):
                col = attr_cols_ind[idx % 2]
                with col:
                    if st.checkbox(attr_name, value=True, key=f"interactive_ind_attr_filter_{idx}"):
                        selected_attrs_ind.append(attr_name)

            show_all_indicators = st.checkbox(
                "Mostrar todos los indicadores",
                value=True,
                key="interactive_ind_show_all",
            )

            if show_all_indicators:
                selected_indicators = all_indicator_names
            else:
                st.markdown("**Selecciona indicadores**")
                indicator_cols = st.columns(2)
                selected_indicators = []
                for idx, indicator_name in enumerate(all_indicator_names):
                    col = indicator_cols[idx % 2]
                    with col:
                        if st.checkbox(indicator_name, value=True, key=f"interactive_ind_name_filter_{idx}"):
                            selected_indicators.append(indicator_name)

            empalmar_indicadores = st.toggle(
                "Empalmar atributos en indicadores",
                value=False,
                key="overlay_indicators_by_attribute",
            )

            df_indicator_filtered = df_indicator_plot[
                df_indicator_plot["Atributo"].isin(selected_attrs_ind)
                & df_indicator_plot["Indicador"].isin(selected_indicators)
            ].copy()

            if df_indicator_filtered.empty:
                st.info("No hay indicadores para mostrar con los filtros seleccionados.")
            else:
                df_ind_sorted = df_indicator_filtered.sort_values(["Atributo", "OrdenGlobal"]).copy()
                x_col_ind = "PosicionAtributo" if empalmar_indicadores else "OrdenGlobal"
                fig_ind_interactive = px.line(
                    df_ind_sorted,
                    x=x_col_ind,
                    y="Valor",
                    color="Atributo",
                    markers=False,
                    hover_name="Indicador",
                    hover_data={"Atributo": True, "Valor": ":.1f", "OrdenGlobal": False, "PosicionAtributo": empalmar_indicadores},
                    color_discrete_map=attribute_colors,
                    category_orders={"Atributo": all_attrs},
                )
                fig_ind_interactive.update_traces(
                    mode="lines",
                    line=dict(width=3),
                    hovertemplate="<b>%{hovertext}</b><br>Atributo: %{fullData.name}<br>Valor: %{y:.1f}<extra></extra>",
                )
                fig_ind_interactive.update_layout(
                    title="Indicadores empalmados por atributo" if empalmar_indicadores else "Indicadores por atributo",
                    xaxis_title="Posición dentro del atributo" if empalmar_indicadores else "Indicadores (ordenados)",
                    yaxis_title="Puntaje",
                    yaxis=dict(range=[0, 100]),
                    clickmode="event+select",
                    hovermode="closest",
                    legend_title_text="Atributo",
                    height=360,
                    margin=dict(l=20, r=20, t=50, b=40),
                )
                if empalmar_indicadores:
                    max_pos_ind = int(df_ind_sorted["PosicionAtributo"].max()) if not df_ind_sorted.empty else 1
                    fig_ind_interactive.update_xaxes(tickmode="array", tickvals=list(range(1, max_pos_ind + 1)), ticktext=[str(i) for i in range(1, max_pos_ind + 1)])
                else:
                    fig_ind_interactive.update_xaxes(
                        tickmode="array",
                        tickvals=df_ind_sorted["OrdenGlobal"].tolist(),
                        ticktext=df_ind_sorted["Indicador"].tolist(),
                        tickangle=-35,
                    )
                st.plotly_chart(fig_ind_interactive, use_container_width=True)
                if empalmar_indicadores:
                    st.caption("Empalme activo: cada línea representa un atributo y la comparación se hace por posición interna. Pasa el cursor para ver nombre real del indicador.")
                else:
                    st.caption("Pasa el cursor sobre la línea para ver el nombre del indicador y su valor real.")

                top_ind = df_indicator_filtered.nlargest(min(3, len(df_indicator_filtered)), "Valor")
                bottom_ind = df_indicator_filtered.nsmallest(min(3, len(df_indicator_filtered)), "Valor")
                fortalezas_ind = ", ".join([f"{r.Indicador} ({r.Valor:.1f})" for r in top_ind.itertuples()])
                debilidades_ind = ", ".join([f"{r.Indicador} ({r.Valor:.1f})" for r in bottom_ind.itertuples()])

                st.markdown("#### Interpretación (indicadores)")
                st.write(
                    f"Las principales fortalezas observadas son: {fortalezas_ind}. "
                    f"Las principales debilidades son: {debilidades_ind}."
                )
                st.write(
                    "**Recomendación:** mantener y documentar las prácticas de los indicadores altos, "
                    "y enfocar acciones rápidas y medibles en los indicadores bajos para cerrar brechas entre atributos."
                )

    st.markdown("---")
    st.markdown("### Intangibles por atributo")
    if not PLOTLY_AVAILABLE:
        detail = f" Detalle técnico: {PLOTLY_ERROR}" if PLOTLY_ERROR else ""
        st.warning(f"No se pudo cargar Plotly para visualizaciones interactivas.{detail}")
    else:
        intang_rows = []
        intang_global_order = 1
        for sector_name, intang_map in [
            ("Encuentro", intangibles_A1),
            ("Conexiones y Accesos", intangibles_A2),
            ("Comodidad e Imagen", intangibles_A3),
            ("Usos y Actividades", intangibles_A4),
        ]:
            local_pos = 1
            for intang_name, intang_value in intang_map.items():
                if intang_value is None:
                    local_pos += 1
                    continue
                intang_rows.append(
                    {
                        "Atributo": sector_name,
                        "Intangible": intang_name,
                        "Valor": float(intang_value),
                        "OrdenGlobal": intang_global_order,
                        "PosicionAtributo": local_pos,
                    }
                )
                intang_global_order += 1
                local_pos += 1

        if intang_rows:
            df_intang_plot = pd.DataFrame(intang_rows)
            all_attrs_int = ["Encuentro", "Conexiones y Accesos", "Comodidad e Imagen", "Usos y Actividades"]
            all_intang_names = df_intang_plot["Intangible"].tolist()

            st.markdown("**Filtrar atributos (intangibles)**")
            attr_cols_int = st.columns(2)
            selected_attrs_int = []
            for idx, attr_name in enumerate(all_attrs_int):
                col = attr_cols_int[idx % 2]
                with col:
                    if st.checkbox(attr_name, value=True, key=f"interactive_int_attr_filter_{idx}"):
                        selected_attrs_int.append(attr_name)

            show_all_intangibles = st.checkbox(
                "Mostrar todos los intangibles",
                value=True,
                key="interactive_int_show_all",
            )

            if show_all_intangibles:
                selected_intang = all_intang_names
            else:
                st.markdown("**Selecciona intangibles**")
                intang_cols = st.columns(2)
                selected_intang = []
                for idx, intang_name in enumerate(all_intang_names):
                    col = intang_cols[idx % 2]
                    with col:
                        if st.checkbox(intang_name, value=True, key=f"interactive_int_name_filter_{idx}"):
                            selected_intang.append(intang_name)

            empalmar_intangibles = st.toggle(
                "Empalmar atributos en intangibles",
                value=False,
                key="overlay_intangibles_by_attribute",
            )

            df_intang_filtered = df_intang_plot[
                df_intang_plot["Atributo"].isin(selected_attrs_int)
                & df_intang_plot["Intangible"].isin(selected_intang)
            ].copy()

            if df_intang_filtered.empty:
                st.info("No hay intangibles para mostrar con los filtros seleccionados.")
            else:
                df_int_sorted = df_intang_filtered.sort_values(["Atributo", "OrdenGlobal"]).copy()
                x_col_int = "PosicionAtributo" if empalmar_intangibles else "OrdenGlobal"
                fig_int_interactive = px.line(
                    df_int_sorted,
                    x=x_col_int,
                    y="Valor",
                    color="Atributo",
                    markers=False,
                    hover_name="Intangible",
                    hover_data={"Atributo": True, "Valor": ":.1f", "OrdenGlobal": False, "PosicionAtributo": empalmar_intangibles},
                    color_discrete_map=attribute_colors,
                    category_orders={"Atributo": all_attrs_int},
                )
                fig_int_interactive.update_traces(
                    mode="lines",
                    line=dict(width=3),
                    hovertemplate="<b>%{hovertext}</b><br>Atributo: %{fullData.name}<br>Valor: %{y:.1f}<extra></extra>",
                )
                fig_int_interactive.update_layout(
                    title="Intangibles empalmados por atributo" if empalmar_intangibles else "Intangibles por atributo",
                    xaxis_title="Posición dentro del atributo" if empalmar_intangibles else "Intangibles (ordenados)",
                    yaxis_title="Puntaje",
                    yaxis=dict(range=[0, 100]),
                    clickmode="event+select",
                    hovermode="closest",
                    legend_title_text="Atributo",
                    height=360,
                    margin=dict(l=20, r=20, t=50, b=40),
                )
                if empalmar_intangibles:
                    max_pos_int = int(df_int_sorted["PosicionAtributo"].max()) if not df_int_sorted.empty else 1
                    fig_int_interactive.update_xaxes(tickmode="array", tickvals=list(range(1, max_pos_int + 1)), ticktext=[str(i) for i in range(1, max_pos_int + 1)])
                else:
                    fig_int_interactive.update_xaxes(
                        tickmode="array",
                        tickvals=df_int_sorted["OrdenGlobal"].tolist(),
                        ticktext=df_int_sorted["Intangible"].tolist(),
                        tickangle=-35,
                    )
                st.plotly_chart(fig_int_interactive, use_container_width=True)
                if empalmar_intangibles:
                    st.caption("Empalme activo: cada línea representa un atributo y la comparación se hace por posición interna. Pasa el cursor para ver nombre real del intangible.")
                else:
                    st.caption("Pasa el cursor sobre la línea para ver el nombre del intangible y su valor real.")

                top_int = df_intang_filtered.nlargest(min(3, len(df_intang_filtered)), "Valor")
                bottom_int = df_intang_filtered.nsmallest(min(3, len(df_intang_filtered)), "Valor")
                fortalezas_int = ", ".join([f"{r.Intangible} ({r.Valor:.1f})" for r in top_int.itertuples()])
                debilidades_int = ", ".join([f"{r.Intangible} ({r.Valor:.1f})" for r in bottom_int.itertuples()])

                st.markdown("#### Interpretación (intangibles)")
                st.write(
                    f"Las principales fortalezas observadas son: {fortalezas_int}. "
                    f"Las principales debilidades son: {debilidades_int}."
                )
                st.write(
                    "**Recomendación:** consolidar los intangibles fuertes con acciones de continuidad comunitaria, "
                    "y priorizar mejoras tácticas en los intangibles bajos para elevar percepción, uso y apropiación del lugar."
                )

    # ===== Interpretación de resultados will be placed after the main diagram =====
    # Define helper function for performance levels
    def nivel_desempeno(score_0_100: float) -> str:
        """Clasifica un puntaje de 0-100 en niveles de desempeño"""
        fraccion = score_0_100 / 100.0
        if fraccion >= 0.85:
            return "excelente"
        elif fraccion >= 0.65:
            return "bueno"
        elif fraccion >= 0.4:
            return "en desarrollo"
        else:
            return "crítico"

    nivel_global = nivel_desempeno(global_score)

    # Orden de atributos para la interpretación
    atributos_interpretacion = [
        ("Encuentro", A1_total),
        ("Conexiones y Accesos", A2_total),
        ("Comodidad e Imagen", A3_total),
        ("Usos y Actividades", A4_total),
    ]

    for nombre_atributo, puntaje in atributos_interpretacion:
        nivel = nivel_desempeno(puntaje)

        if nombre_atributo == "Encuentro":
            if nivel == "excelente":
                texto = (
                    f"En **Encuentro** obtuviste {puntaje:.1f} de 100 puntos, lo que refleja "
                    "un nivel excelente. Tu lugar facilita que las personas se vean, saluden "
                    "a vecinos, convivan con familias, niñas, niños y personas mayores, y se "
                    "sienta un fuerte orgullo y sentido de pertenencia. Es un espacio donde "
                    "la comunidad realmente se reconoce y se encuentra."
                )
            elif nivel == "bueno":
                texto = (
                    f"En **Encuentro** obtuviste {puntaje:.1f} de 100 puntos. "
                    "Tu lugar funciona como un punto de reunión aceptable: hay presencia de "
                    "familias y grupos diversos, y cierta organización vecinal, aunque aún se "
                    "podría fortalecer la participación comunitaria y la sensación de orgullo "
                    "y apropiación del lugar."
                )
            elif nivel == "en desarrollo":
                texto = (
                    f"En **Encuentro** obtuviste {puntaje:.1f} de 100 puntos. "
                    "El sitio ofrece algunas oportunidades de interacción, pero estas son "
                    "limitadas o esporádicas. Podría haber poca evidencia de vecinos "
                    "organizados o de actividades que integren a niñas, niños y personas "
                    "mayores, lo que reduce el apego al lugar."
                )
            else:
                texto = (
                    f"En **Encuentro** obtuviste {puntaje:.1f} de 100 puntos, lo que indica "
                    "un nivel crítico. Es probable que el lugar no se use como punto de "
                    "reunión, que casi no haya interacción entre vecinos y que los grupos "
                    "demográficos (niñas, niños, mayores, familias) estén poco presentes. "
                    "Esto limita fuertemente el sentido de comunidad en el espacio."
                )

        elif nombre_atributo == "Usos y Actividades":
            if nivel == "excelente":
                texto = (
                    f"En **Usos y Actividades** obtuviste {puntaje:.1f} de 100 puntos. "
                    "Tu lugar ofrece muchas opciones de actividades, las personas se quedan "
                    "tiempo, las niñas y niños se ven entretenidos y el mobiliario resulta "
                    "funcional. Además, la mezcla de comercios y servicios y la vitalidad "
                    "económica hacen que siempre haya algo que hacer."
                )
            elif nivel == "bueno":
                texto = (
                    f"En **Usos y Actividades** obtuviste {puntaje:.1f} de 100 puntos. "
                    "El sitio tiene varias actividades y usos, pero todavía hay momentos o "
                    "zonas donde no pasa mucho. La oferta de mobiliario, comercio o servicios "
                    "es adecuada, aunque podría diversificarse para atraer a más personas "
                    "y prolongar su permanencia."
                )
            elif nivel == "en desarrollo":
                texto = (
                    f"En **Usos y Actividades** obtuviste {puntaje:.1f} de 100 puntos. "
                    "Probablemente hay pocas opciones claras de actividad y el espacio se "
                    "percibe más como un lugar de paso que de estancia. Esto hace que el sitio "
                    "luzca vacío en ciertos momentos y que el mobiliario o los servicios no "
                    "estén aprovechados."
                )
            else:
                texto = (
                    f"En **Usos y Actividades** obtuviste {puntaje:.1f} de 100 puntos, "
                    "indicando un nivel crítico. Casi no hay razones para quedarse en el lugar: "
                    "faltan actividades, servicios atractivos o mobiliario útil. "
                    "En estas condiciones, el espacio tiende a permanecer vacío y poco visible "
                    "para la comunidad."
                )

        elif nombre_atributo == "Conexiones y Accesos":
            if nivel == "excelente":
                texto = (
                    f"En **Conexiones y Accesos** obtuviste {puntaje:.1f} de 100 puntos. "
                    "Tu lugar es fácil de alcanzar, caminar y recorrer: está bien conectado "
                    "con su entorno, tiene paradas de transporte público cercanas, "
                    "rampas accesibles y señalización clara. Esto facilita que muchas personas "
                    "lo usen diariamente."
                )
            elif nivel == "bueno":
                texto = (
                    f"En **Conexiones y Accesos** obtuviste {puntaje:.1f} de 100 puntos. "
                    "En general, el lugar es accesible y visible, pero puede haber ciertos "
                    "tramos incómodos para caminar, falta de rampas en algunos puntos o "
                    "señalización que podría ser más clara. Aun así, la mayoría de las "
                    "personas puede llegar sin demasiadas dificultades."
                )
            elif nivel == "en desarrollo":
                texto = (
                    f"En **Conexiones y Accesos** obtuviste {puntaje:.1f} de 100 puntos. "
                    "El sitio no siempre resulta fácil de alcanzar o atravesar: quizá la "
                    "ubicación no es tan conveniente, la caminabilidad es limitada o la "
                    "conexión con el transporte público es débil. Esto reduce el flujo de "
                    "personas que pueden disfrutar del lugar."
                )
            else:
                texto = (
                    f"En **Conexiones y Accesos** obtuviste {puntaje:.1f} de 100 puntos, "
                    "lo que señala un nivel crítico. Es probable que llegar al lugar sea "
                    "difícil, que no existan buenas rutas peatonales ni rampas, y que la "
                    "señalización sea escasa o confusa. Todo esto hace que el sitio parezca "
                    "aislado o poco visible."
                )

        else:  # Comodidad e Imagen
            if nivel == "excelente":
                texto = (
                    f"En **Comodidad e Imagen** obtuviste {puntaje:.1f} de 100 puntos. "
                    "El lugar se percibe atractivo, agradable y seguro; está limpio, bien "
                    "mantenido y cuenta con suficientes lugares cómodos para sentarse. "
                    "La gente se siente a gusto permaneciendo ahí."
                )
            elif nivel == "bueno":
                texto = (
                    f"En **Comodidad e Imagen** obtuviste {puntaje:.1f} de 100 puntos. "
                    "La imagen general del lugar es positiva, aunque es posible que haya "
                    "detalles de mantenimiento, limpieza o cantidad de asientos que podrían "
                    "mejorarse para aumentar la sensación de confort y seguridad."
                )
            elif nivel == "en desarrollo":
                texto = (
                    f"En **Comodidad e Imagen** obtuviste {puntaje:.1f} de 100 puntos. "
                    "Es probable que el sitio presente cierta incomodidad: pocos asientos, "
                    "áreas poco agradables o percepción de inseguridad en ciertos horarios. "
                    "La limpieza y el mantenimiento podrían no ser constantes."
                )
            else:
                texto = (
                    f"En **Comodidad e Imagen** obtuviste {puntaje:.1f} de 100 puntos, "
                    "lo que indica un nivel crítico. El lugar puede percibirse sucio, "
                    "descuidado o inseguro, con escasos sitios para sentarse y poca "
                    "sensación de confort. Esto desincentiva que las personas permanezcan."
                )

        st.write(texto)

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

    # Mostrar áreas de mejora solo para atributos en desarrollo o críticos
    areas_mejora = []
    for nombre_atributo, puntaje in atributos_interpretacion:
        nivel = nivel_desempeno(puntaje)
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
            areas_mejora.append(msg)

    if not areas_mejora:
        st.write("¡Tu lugar está funcionando muy bien en todos los atributos! Continúa con el buen trabajo de mantenimiento y activación comunitaria.")

    # d) Cómo puede mejorar su lugar (recomendaciones)
    st.subheader("¿Cómo puedes mejorar tu lugar?")
    st.write(
        "A partir de tu diagnóstico, puedes plantear un plan de mejora gradual. "
        "Te sugerimos priorizar acciones de bajo costo y alto impacto, involucrando "
        "a la comunidad desde el inicio:"
    )

    for nombre_atributo, puntaje in atributos_interpretacion:
        nivel = nivel_desempeno(puntaje)

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

    # ===== INTERPRETACIÓN GENERAL (A): Después del diagrama =====
    st.markdown("---")
    st.markdown('<div id="analisis_resultados"></div>', unsafe_allow_html=True)
    st.header("Interpretación de tus resultados")

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

    # ==================== Vista detallada por atributo (CUADRANTE) ====================
    st.markdown("### Vista detallada por atributo")
    selected_sector = st.selectbox("Selecciona un atributo", sector_names, index=0)

    # Etiquetas "humanas" de los indicadores (anillo exterior)
    INDICATOR_DISPLAY_LABELS = {
        "Encuentro": [
            "Diversidad\ndemográfica",
            "Redes\nciudadanas",
            "Voluntariado",
            "Cuidado\ndel lugar",
            "Uso\nnocturno",
        ],
        "Conexiones y Accesos": [
            "Modos de\ntransporte",
            "Conectividad\ncon el lugar",
            "Permanencia",
            "Accesibilidad\ndel entorno",
            "Accesibilidad\ndentro del lugar\n(movilidad reducida)",
            "Accesibilidad\ndentro del lugar (primera\ninfancia y cuidadores)",
        ],
        "Comodidad e Imagen": [
            "Seguridad\ny limpieza",
            "Cuidado de\nla imagen",
            "Comodidad",
            "Caminabilidad",
            "Lugares para\nsentarse",
            "Resiliencia\nclimática y\náreas verdes",
            "Ser agradable",
        ],
        "Usos y Actividades": [
            "Dinamismo",
            "Referente",
            "Utilidad",
            "Actividad\neconómica",
            "Diversidad de\nactividades",
            "Localidad",
        ],
    }

    # Ángulos manuales
    MIDDLE_ANGLES = {
        "Encuentro": {
            "Diversidad": 86,
            "Cuidado": 74,
            "Comunidad": 62,
            "Compartido": 51,
            "Símbolos": 40,
            "Orgullo": 29,
            "Amigable": 16,
            "Interactivo": 4,
        },
        "Conexiones y Accesos": {
            "Cercano": 85,
            "Conectado": 68,
            "Conveniente": 53,
            "Accesible\n(movilidad reducida)": 37.5,
            "Accesible (primera\ninfancia y cuidadores)": 22.8,
            "Transitable": 6,
        },
        "Comodidad e Imagen": {
            "Limpio": 84,
            "Seguro": 72,
            "Sentable": 59,
            "Agradable": 45,
            "Verde": 30,
            "Caminable": 16,
            "Resiliencia climática": 5,
        },
        "Usos y Actividades": {
            "Dinámico": 85,
            "Especial": 75,
            "Real": 65,
            "Útil": 55,
            "Local": 33,
            "Sostenible": 24,
            "Conmemorativo": 15,
            "Pertenencia": 5,
        },
    }

    OUTER_ANGLES = {
        "Encuentro": {
            "Diversidad\ndemográfica": 81,
            "Redes\nciudadanas": 63,
            "Voluntariado": 44.5,
            "Cuidado\ndel lugar": 27,
            "Uso\nnocturno": 8.5,
        },
        "Conexiones y Accesos": {
            "Modos de\ntransporte": 82,
            "Conectividad\ncon el lugar": 68,
            "Permanencia": 53,
            "Accesibilidad\ndel entorno": 37.5,
            "Accesibilidad\ndentro del lugar\n(movilidad reducida)": 22.8,
            "Accesibilidad\ndentro del lugar (primera\ninfancia y cuidadores)": 8,
        },
        "Comodidad e Imagen": {
            "Seguridad\ny limpieza": 83,
            "Cuidado de\nla imagen": 70,
            "Comodidad": 58,
            "Caminabilidad": 45,
            "Lugares para\nsentarse": 32,
            "Resiliencia\nclimática y\náreas verdes": 19.5,
            "Ser agradable": 7,
        },
        "Usos y Actividades": {
            "Dinamismo": 84.3,
            "Referente": 70,
            "Utilidad": 52,
            "Actividad\neconómica": 37,
            "Diversidad de\nactividades": 20,
            "Localidad": 6,
        },
    }

    SECTOR_TO_INDICATORS = {
        "Encuentro": A1_indicators,
        "Conexiones y Accesos": A2_indicators,
        "Comodidad e Imagen": A3_indicators,
        "Usos y Actividades": A4_indicators,
    }
    SECTOR_TO_INTANGIBLES = {
        "Encuentro": intangibles_A1,
        "Conexiones y Accesos": intangibles_A2,
        "Comodidad e Imagen": intangibles_A3,
        "Usos y Actividades": intangibles_A4,
    }

    middle_pcts = sector_scores[selected_sector]["middle_pcts"]
    outer_pcts = sector_scores[selected_sector]["outer_pcts"]
    mid_labels = SECTOR_CONFIG[selected_sector]["middle_labels"]
    human_outer_labels = INDICATOR_DISPLAY_LABELS[selected_sector]
    colors = track_colors[selected_sector]

    # Configuración del cuadrante (0° – 90°)
    theta_start = 0.0
    theta_end = np.pi / 2.0
    theta_range = theta_end - theta_start
    r_inner_min_q, r_inner_max_q = 0.00, 0.17
    r_mid_min_q, r_mid_max_q = 0.17, 0.53
    r_outer_min_q, r_outer_max_q = 0.53, 0.95

    fig_q, ax_q = plt.subplots(
        figsize=(4.5, 4.5),
        subplot_kw={"projection": "polar"},
    )
    ax_q.set_theta_offset(np.pi / 2.0)
    ax_q.set_theta_direction(-1)
    ax_q.set_thetamin(0)
    ax_q.set_thetamax(90)
    ax_q.set_ylim(0, 1.0)
    ax_q.spines["polar"].set_visible(False)
    ax_q.set_frame_on(False)
    ax_q.patch.set_visible(False)
    ax_q.grid(False)
    ax_q.set_xticks([])
    ax_q.set_yticks([])

    def get_angle(sector: str, label: str, idx: int, default_list: list) -> float:
        if sector in MIDDLE_ANGLES and label in MIDDLE_ANGLES[sector]:
            return MIDDLE_ANGLES[sector][label]
        if sector in OUTER_ANGLES and label in OUTER_ANGLES[sector]:
            return OUTER_ANGLES[sector][label]
        if idx < len(default_list):
            return default_list[idx]
        return 0.0

    # Anillo interior (fondo del atributo)
    theta_center_attr = (theta_start + theta_end) / 2.0
    ax_q.bar(
        theta_center_attr,
        r_inner_max_q - r_inner_min_q,
        width=theta_range,
        bottom=r_inner_min_q,
        color=colors["inner"],
        edgecolor="white",
        linewidth=8,
        align="center",
    )

    # ANILLO MIDDLE (intangibles)
    middle_blocks = len(middle_pcts)
    if middle_blocks > 0:
        theta_block_mid = theta_range / middle_blocks
        height_mid = r_mid_max_q - r_mid_min_q
        r_mid_label = r_mid_min_q + 0.06
        default_mid_angles = [
            9,
            27,
            45,
            63,
            81,
            6.4,
            19.3,
            32.1,
            45,
            57.9,
            70.7,
            83.6,
        ]
        for i, pct in enumerate(middle_pcts):
            ratio = nz(pct) / 100.0
            theta_center = theta_start + (i + 0.5) * theta_block_mid
            if ratio > 0:
                ax_q.bar(
                    theta_center,
                    height_mid * ratio,
                    width=theta_block_mid,
                    bottom=r_mid_min_q,
                    color=colors["middle"],
                    edgecolor="white",
                    linewidth=5,
                    align="center",
                )
            if ratio < 1:
                ax_q.bar(
                    theta_center,
                    height_mid * (1 - ratio),
                    width=theta_block_mid,
                    bottom=r_mid_min_q + height_mid * ratio,
                    color="white",
                    edgecolor="white",
                    linewidth=5,
                    align="center",
                )
            etiqueta = mid_labels[i]
            rot = get_angle(selected_sector, etiqueta, i, default_mid_angles)
            ax_q.text(
                theta_center,
                r_mid_label,
                etiqueta,
                fontsize=6,
                fontname="Poppins",
                rotation=rot,
                rotation_mode="anchor",
                ha="left",
                va="center",
            )

    # ANILLO OUTER (indicadores)
    outer_blocks = len(outer_pcts)
    if outer_blocks > 0:
        theta_block_out = theta_range / outer_blocks
        height_out = r_outer_max_q - r_outer_min_q
        r_outer_label = r_outer_min_q + 0.02
        default_outer_angles = [
            9,
            27,
            45,
            63,
            81,
            7.5,
            22.5,
            37.5,
            52.5,
            67.5,
            82.5,
        ]
        for i, pct in enumerate(outer_pcts):
            ratio = nz(pct) / 100.0
            theta_center = theta_start + (i + 0.5) * theta_block_out
            if ratio > 0:
                ax_q.bar(
                    theta_center,
                    height_out * ratio,
                    width=theta_block_out,
                    bottom=r_outer_min_q,
                    color=colors["outer"],
                    edgecolor="white",
                    linewidth=5,
                    align="center",
                )
            if ratio < 1:
                ax_q.bar(
                    theta_center,
                    height_out * (1 - ratio),
                    width=theta_block_out,
                    bottom=r_outer_min_q + height_out * ratio,
                    color="white",
                    edgecolor="white",
                    linewidth=5,
                    align="center",
                )
            etiqueta = human_outer_labels[i]
            rot = get_angle(selected_sector, etiqueta, i, default_outer_angles)
            ax_q.text(
                theta_center,
                r_outer_label,
                etiqueta,
                fontsize=7,
                fontname="Poppins",
                rotation=rot,
                rotation_mode="anchor",
                ha="left",
                va="center",
            )

    st.pyplot(fig_q, use_container_width=True)

    # ===== INTERPRETACIÓN POR ATRIBUTO (B): Después del diagrama detallado =====
    st.markdown("---")

    # Determinar el puntaje del atributo seleccionado
    atributos_interpretacion = [
        ("Encuentro", A1_total),
        ("Conexiones y Accesos", A2_total),
        ("Comodidad e Imagen", A3_total),
        ("Usos y Actividades", A4_total),
    ]

    puntaje_seleccionado = next((puntaje for nombre, puntaje in atributos_interpretacion if nombre == selected_sector), 0)
    nivel_seleccionado = nivel_desempeno(puntaje_seleccionado)

    # Generar explicación contextualizada para el atributo seleccionado
    st.subheader(f"¿Qué significa tu resultado en {selected_sector}?")

    if selected_sector == "Encuentro":
        if nivel_seleccionado == "excelente":
            texto = (
                f"En **Encuentro** obtuviste {puntaje_seleccionado:.1f} de 100 puntos, lo que refleja "
                "un nivel excelente. Tu lugar facilita que las personas se vean, saluden "
                "a vecinos, convivan con familias, niñas, niños y personas mayores, y se "
                "sienta un fuerte orgullo y sentido de pertenencia. Es un espacio donde "
                "la comunidad realmente se reconoce y se encuentra."
            )
        elif nivel_seleccionado == "bueno":
            texto = (
                f"En **Encuentro** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "Tu lugar funciona como un punto de reunión aceptable: hay presencia de "
                "familias y grupos diversos, y cierta organización vecinal, aunque aún se "
                "podría fortalecer la participación comunitaria y la sensación de orgullo "
                "y apropiación del lugar."
            )
        elif nivel_seleccionado == "en desarrollo":
            texto = (
                f"En **Encuentro** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "El sitio ofrece algunas oportunidades de interacción, pero estas son "
                "limitadas o esporádicas. Podría haber poca evidencia de vecinos "
                "organizados o de actividades que integren a niñas, niños y personas "
                "mayores, lo que reduce el apego al lugar."
            )
        else:
            texto = (
                f"En **Encuentro** obtuviste {puntaje_seleccionado:.1f} de 100 puntos, lo que indica "
                "un nivel crítico. Es probable que el lugar no se use como punto de "
                "reunión, que casi no haya interacción entre vecinos y que los grupos "
                "demográficos (niñas, niños, mayores, familias) estén poco presentes. "
                "Esto limita fuertemente el sentido de comunidad en el espacio."
            )

    elif selected_sector == "Usos y Actividades":
        if nivel_seleccionado == "excelente":
            texto = (
                f"En **Usos y Actividades** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "Tu lugar ofrece muchas opciones de actividades, las personas se quedan "
                "tiempo, las niñas y niños se ven entretenidos y el mobiliario resulta "
                "funcional. Además, la mezcla de comercios y servicios y la vitalidad "
                "económica hacen que siempre haya algo que hacer."
            )
        elif nivel_seleccionado == "bueno":
            texto = (
                f"En **Usos y Actividades** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "El sitio tiene varias actividades y usos, pero todavía hay momentos o "
                "zonas donde no pasa mucho. La oferta de mobiliario, comercio o servicios "
                "es adecuada, aunque podría diversificarse para atraer a más personas "
                "y prolongar su permanencia."
            )
        elif nivel_seleccionado == "en desarrollo":
            texto = (
                f"En **Usos y Actividades** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "Probablemente hay pocas opciones claras de actividad y el espacio se "
                "percibe más como un lugar de paso que de estancia. Esto hace que el sitio "
                "luzca vacío en ciertos momentos y que el mobiliario o los servicios no "
                "estén aprovechados."
            )
        else:
            texto = (
                f"En **Usos y Actividades** obtuviste {puntaje_seleccionado:.1f} de 100 puntos, "
                "indicando un nivel crítico. Casi no hay razones para quedarse en el lugar: "
                "faltan actividades, servicios atractivos o mobiliario útil. "
                "En estas condiciones, el espacio tiende a permanecer vacío y poco visible "
                "para la comunidad."
            )

    elif selected_sector == "Conexiones y Accesos":
        if nivel_seleccionado == "excelente":
            texto = (
                f"En **Conexiones y Accesos** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "Tu lugar es fácil de alcanzar, caminar y recorrer: está bien conectado "
                "con su entorno, tiene paradas de transporte público cercanas, "
                "rampas accesibles y señalización clara. Esto facilita que muchas personas "
                "lo usen diariamente."
            )
        elif nivel_seleccionado == "bueno":
            texto = (
                f"En **Conexiones y Accesos** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "En general, el lugar es accesible y visible, pero puede haber ciertos "
                "tramos incómodos para caminar, falta de rampas en algunos puntos o "
                "señalización que podría ser más clara. Aun así, la mayoría de las "
                "personas puede llegar sin demasiadas dificultades."
            )
        elif nivel_seleccionado == "en desarrollo":
            texto = (
                f"En **Conexiones y Accesos** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "El sitio no siempre resulta fácil de alcanzar o atravesar: quizá la "
                "ubicación no es tan conveniente, la caminabilidad es limitada o la "
                "conexión con el transporte público es débil. Esto reduce el flujo de "
                "personas que pueden disfrutar del lugar."
            )
        else:
            texto = (
                f"En **Conexiones y Accesos** obtuviste {puntaje_seleccionado:.1f} de 100 puntos, "
                "lo que señala un nivel crítico. Es probable que llegar al lugar sea "
                "difícil, que no existan buenas rutas peatonales ni rampas, y que la "
                "señalización sea escasa o confusa. Todo esto hace que el sitio parezca "
                "aislado o poco visible."
            )

    else:  # Comodidad e Imagen
        if nivel_seleccionado == "excelente":
            texto = (
                f"En **Comodidad e Imagen** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "El lugar se percibe atractivo, agradable y seguro; está limpio, bien "
                "mantenido y cuenta con suficientes lugares cómodos para sentarse. "
                "La gente se siente a gusto permaneciendo ahí."
            )
        elif nivel_seleccionado == "bueno":
            texto = (
                f"En **Comodidad e Imagen** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "La imagen general del lugar es positiva, aunque es posible que haya "
                "detalles de mantenimiento, limpieza o cantidad de asientos que podrían "
                "mejorarse para aumentar la sensación de confort y seguridad."
            )
        elif nivel_seleccionado == "en desarrollo":
            texto = (
                f"En **Comodidad e Imagen** obtuviste {puntaje_seleccionado:.1f} de 100 puntos. "
                "Es probable que el sitio presente cierta incomodidad: pocos asientos, "
                "áreas poco agradables o percepción de inseguridad en ciertos horarios. "
                "La limpieza y el mantenimiento podrían no ser constantes."
            )
        else:
            texto = (
                f"En **Comodidad e Imagen** obtuviste {puntaje_seleccionado:.1f} de 100 puntos, "
                "lo que indica un nivel crítico. El lugar puede percibirse sucio, "
                "descuidado o inseguro, con escasos sitios para sentarse y poca "
                "sensación de confort. Esto desincentiva que las personas permanezcan."
            )

    st.write(texto)

    # Tablas de indicadores e intangibles
    st.markdown("#### Indicadores del atributo seleccionado")
    indicadores_dict = SECTOR_TO_INDICATORS[selected_sector]
    indicador_labels = INDICATOR_DISPLAY_LABELS[selected_sector]
    outer_codes = SECTOR_CONFIG[selected_sector]["outer_labels"]
    ind_vals = [indicadores_dict.get(code) for code in outer_codes]
    df_indicadores = pd.DataFrame(
        {
            "Indicador": indicador_labels,
            "Valor (0–100)": [format_score_with_na(v) for v in ind_vals],
        }
    )
    st.table(df_indicadores)

    st.markdown("#### Intangibles del atributo seleccionado")
    intang_dict = SECTOR_TO_INTANGIBLES[selected_sector]
    intangible_names = list(intang_dict.keys())
    intangible_vals = [intang_dict[k] for k in intangible_names]
    df_intang = pd.DataFrame(
        {
            "Intangible": intangible_names,
            "Valor (0–100)": [format_score_with_na(v) for v in intangible_vals],
        }
    )
    st.table(df_intang)

    # --- Exportar CSV ---
    st.markdown("---")
    st.subheader("Descargar resultados")
    data = {
        "nombre_lugar": nombre_lugar,
        "nombre_evaluador": nombre_eval,
        "programa": program_id,
        "genero_id": ans.get("A0_1"),
        "equipo_responsable_id": A0_3,
        "A1.1": A1_indicators["A1.1"],
        "A1.2": A1_indicators["A1.2"],
        "A1.3": A1_indicators["A1.3"],
        "A1.4": A1_indicators["A1.4"],
        "A1.5": A1_indicators["A1.5"],
        "A1_total": A1_total,
        "A2.1": A2_indicators["A2.1"],
        "A2.2": A2_indicators["A2.2"],
        "A2.3": A2_indicators["A2.3"],
        "A2.4": A2_indicators["A2.4"],
        "A2.5": A2_indicators["A2.5"],
        "A2.6": A2_indicators["A2.6"],
        "A2_total": A2_total,
        "A3.1": A3_indicators["A3.1"],
        "A3.2": A3_indicators["A3.2"],
        "A3.3": A3_indicators["A3.3"],
        "A3.4": A3_indicators["A3.4"],
        "A3.5": A3_indicators["A3.5"],
        "A3.6": A3_indicators["A3.6"],
        "A3.7": A3_indicators["A3.7"],
        "A3_total": A3_total,
        "A4.1": A4_indicators["A4.1"],
        "A4.2": A4_indicators["A4.2"],
        "A4.3": A4_indicators["A4.3"],
        "A4.4": A4_indicators["A4.4"],
        "A4.5": A4_indicators["A4.5"],
        "A4.6": A4_indicators["A4.6"],
        "A4_total": A4_total,
        "global_score": global_score,
    }
    save_pending = st.session_state.get("save_pending", False)
    saved_to_sheet = st.session_state.get("saved_to_sheet", False)
    if save_pending and not saved_to_sheet:
        ok, msg = append_to_google_sheet(data)
        if ok:
            st.session_state["saved_to_sheet"] = True
            st.success("Resultados guardados en Google Sheets.")
        else:
            st.warning(f"No se pudo guardar en Google Sheets: {msg}")
        st.session_state["save_pending"] = False
    elif save_pending and saved_to_sheet:
        st.info("Resultados ya guardados en Google Sheets.")
        st.session_state["save_pending"] = False
    df = pd.DataFrame([data])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    safe_name = (nombre_lugar or "lugar").replace(" ", "_")
    st.download_button(
        "Descargar resultados en CSV",
        data=csv_bytes,
        file_name=f"evaluacion_{safe_name}.csv",
        mime="text/csv",
        )


# =========================================================
# 10. NAVEGACIÓN ENTRE SECCIONES (TABS)
# =========================================================
sections = [
    ("Antes de empezar", pagina_antes),
    ("Encuentro", pagina_A1),
    ("Conexiones y Accesos", pagina_A2),
    ("Comodidad e Imagen", pagina_A3),
    ("Usos y Actividades", pagina_A4),
]
tab_labels = [name for name, _ in sections] + ["Resultados"]
st.markdown(
    "Recorre todas las pestañas de atributos para completar la evaluación y al final revisa tus resultados en la pestaña **Resultados**."
)
st.markdown("<div id='tab_selector'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <style>
    [data-baseweb="tab-list"] {
        gap: 6px;
        flex-wrap: wrap;
        justify-content: flex-start;
    }
    [data-baseweb="tab"] {
        border-radius: 999px !important;
        padding: 6px 12px;
        background: transparent;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: var(--space);
        color: var(--white);
    }
    [data-baseweb="tab"]:not([aria-selected="true"]):hover {
        color: var(--blue);
    }
    [data-baseweb="tab"][aria-selected="true"]:hover {
        color: var(--white);
    }
    [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    .tab-selector-link,
    .tab-selector-link:link,
    .tab-selector-link:visited {
        background-color: var(--space);
        border: 1px solid var(--space);
        color: var(--white) !important;
        border-radius: 0.5rem;
        padding: 0.35rem 0.9rem;
        text-decoration: none !important;
        display: inline-block;
    }
    .tab-selector-link:hover,
    .tab-selector-link:active,
    .tab-selector-link:focus {
        background-color: var(--indigo);
        border-color: var(--indigo);
        color: var(--white) !important;
        text-decoration: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
tabs = st.tabs(tab_labels)

for tab, (_, render_section) in zip(tabs[: len(sections)], sections):
    with tab:
        render_section()

with tabs[-1]:
    st.button(
        "Guardar resultados en Google Sheets",
        on_click=trigger_save_to_sheet,
        key="btn_save_results",
        use_container_width=True,
    )
    pagina_resultados()
    st.button(
        "Evaluar otro lugar",
        on_click=reset_evaluacion,
        key="btn_restart",
        use_container_width=True,
    )

render_floating_back_to_top()
st.markdown(
    "<div style='text-align:center; margin-top:1.25rem; opacity:0.8;'>©FPM - José Bucio 2026</div>",
    unsafe_allow_html=True,
)
