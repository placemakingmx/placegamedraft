import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycirclize import Circos
import streamlit as st
import streamlit.components.v1 as components
from streamlit.errors import StreamlitSecretNotFoundError
import gspread

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
    """Promedio ponderado de indicadores (0–100) de un atributo."""
    if not indicators:
        return 0.0
    weights = get_indicator_weights(program_id, attribute_id, indicators)
    num = 0.0
    den = 0.0
    for ind_id, value in indicators.items():
        if value is None:
            continue
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


def compute_global_score(program_id: str, section_scores: dict) -> float:
    """Score global ponderado por sección según el programa."""
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
    return num / den if den > 0 else 0.0


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
        return score_1_4_100_75_50_25(a13_1)
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
        "Símbolos": A1_2,
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
    S222 = score_1_3_10_50_100(a22_2)
    if a22_3 == 1:
        S223 = 100.0
    elif a22_3 == 2:
        S223 = 50.0
    else:
        S223 = None
    if None in (S221, S222, S223):
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
    if S251 is None and S253 is None and S252 is None:
        return None
    if S252 is None:
        if S251 is None or S253 is None:
            return None
        return 0.75 * S251 + 0.25 * S253
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
        S441 = score_A4_4_1(a44_1)
        S442 = score_A4_4_2(a44_2)
        if S441 is None or S442 is None:
            return None
        return 0.4 * S441 + 0.6 * S442
    else:
        vals = [v for v in [A4_1, A4_2, A4_3, A4_5, A4_6] if v is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)


def calc_intangibles_A4(A4_1, A4_2, A4_3, A4_4, A4_5, A4_6):
    A4_1 = nz(A4_1)
    A4_2 = nz(A4_2)
    A4_3 = nz(A4_3)
    A4_4 = nz(A4_4)
    A4_5 = nz(A4_5)
    A4_6 = nz(A4_6)
    return {
        "Dinámico": 0.34 * A4_1 + 0.33 * A4_5 + 0.33 * A4_6,
        "Especial": 0.5 * A4_1 + 0.5 * A4_2,
        "Real": 0.5 * A4_2 + 0.5 * A4_3,
        "Útil": 0.5 * A4_2 + 0.5 * A4_3,
        "Local": 0.33 * A4_4 + 0.34 * A4_5 + 0.33 * A4_6,
        "Sostenible": A4_4,
        "Conmemorativo": A4_4,
        "Comunidad": A4_6,
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


# =========================================================
# 8. PÁGINAS (PASO A PASO)
# =========================================================
st.title("Diagrama de Lugar")
# ANCLA AL INICIO DE LA APP
st.markdown("<div id='top-of-page'></div>", unsafe_allow_html=True)

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

    if a12_1 == 1:
        # NO hay grupos organizados → solo se pregunta a12_2_2
        clear_branch("a12_2_1")
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
    if a12_1 == 1:
        # Si NO hay redes: usamos el mismo valor de estado físico (a12_2_2) como proxy
        a12_2_2_val = get_ans("a12_2_2")
        set_ans("a13_1", a12_2_2_val)
        set_ans("a14_1", a12_2_2_val)

        # Importante: limpiar posibles widgets no visibles
        clear_branch("a13_1")  # a13_1 NO se captura con widget en esta rama
        # (Pero dejamos el valor en answers, así que lo re-seteamos)
        set_ans("a13_1", a12_2_2_val)

    else:
        st.markdown("**Voluntariado**")

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
        # Si hay uso nocturno, a15_2 no aplica
        clear_branch("a15_2")
        set_ans("a15_2", 1)  # lógica interna: "siempre hay gente"
    else:
        # Si no hay uso nocturno (o no se sabe), a15_2 sí aplica
        if a15_1 in (2, 3):
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
        else:
            # Caso defensivo: si a15_1 es None
            clear_branch("a15_2")
            set_ans("a15_2", None)

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

    if a21_use_percent == 1:
        # Rama con porcentajes → a21_2 no aplica
        clear_branch("a21_2")

        slider_answer("a21_1_walk", "Porcentaje de personas que llega caminando", 0, 100)
        slider_answer("a21_1_bike", "Porcentaje que llega en bicicleta", 0, 100)
        slider_answer("a21_1_pt", "Porcentaje que llega en transporte público", 0, 100)
        slider_answer("a21_1_car", "Porcentaje que llega en auto particular", 0, 100)

    else:
        # Rama sin porcentajes → limpiar sliders + mostrar a21_2
        clear_branch("a21_1_walk", "a21_1_bike", "a21_1_pt", "a21_1_car")

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
    radio_answer("a22_3", "¿Has cambiado a modos más sustentables tras la intervención?", [1, 2], {1: "Sí", 2: "No"})

    # ---------- Permanencia ----------
    radio_answer(
        "a23_1",
        "¿El lugar suele estar lleno?",
        [1, 2, 3, 4],
        {1: "Siempre", 2: "Frecuentemente", 3: "A veces", 4: "Casi vacío"},
    )

    # ---------- Accesibilidad del entorno ----------
    st.markdown("**Accesibilidad del entorno**")
    slider_answer("a24_1", "Accesibilidad para PMR (0-100 %)", 0, 100)
    checkbox_answer("a24_1_na", "No lo sé / Sin datos")

    st.link_button("Calculadora indicadores INEGI", "https://pmm-calculadora-indicadores.streamlit.app/")

    number_answer("a24_2", "Puntaje de Accesibilidad del entorno (0-10)", 0.0, 10.0, 0.1)
    checkbox_answer("a24_2_na", "Sin dato de accesibilidad del entorno")

    # ---------- Accesibilidad dentro del lugar ----------
    st.markdown("**Accesibilidad dentro del lugar**")
    radio_answer(
        "a25_1",
        "¿Hay infraestructura PMR dentro del lugar?",
        [1, 2, 3],
        {1: "Suficiente", 2: "Insuficiente", 3: "No hay"},
    )
    number_answer("a25_2", "Puntaje de Conexión del entorno (0-10)", 0.0, 10.0, 0.1)
    checkbox_answer("a25_2_na", "Sin dato de conexión del entorno")
    radio_answer(
        "a25_3",
        "¿Personas con PMR usan el lugar?",
        [1, 2, 3],
        {1: "Sí", 2: "A veces", 3: "No"},
    )

    # ---------- Primera infancia y cuidadores ----------
    st.markdown("**Accesibilidad para primera infancia y cuidadores**")
    radio_answer(
        "a26_1_p",
        "¿Hay áreas adecuadas para niñez <6 y cuidadores?",
        [1, 2, 3],
        {1: "Sí, de calidad", 2: "Sí, pero limitadas", 3: "No existen"},
    )
    a26_2 = radio_answer("a26_2", "¿Eres cuidador/a actualmente?", [1, 2], {1: "Sí", 2: "No"})
    if a26_2 == 1:
        radio_answer(
            "a26_2_1",
            "Como cuidador/a, ¿qué tan satisfecho estás?",
            [1, 2, 3],
            {1: "Muy satisfecho", 2: "Aceptable", 3: "Insatisfecho"},
        )
    else:
        clear_branch("a26_2_1")
        set_ans("a26_2_1", None)


def pagina_A3():
    st.markdown('<div id="header_comodidad"></div>', unsafe_allow_html=True)
    st.header("Comodidad e imagen")

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
    if A0_3 == 1:
        # Rama equipo responsable → solo a32_2 visible, a32_1 se deriva
        clear_branch("a32_1")

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
        "¿Consideras que el lugar es lo suficientemente verde para ser cómodo a lo largo de todo el año?",
        options=[1, 2, 3],
        labels_map={
            1: "Sí, está genial",
            2: "No está mal, pero podría mejorar",
            3: "Creo que olvidaron plantar al menos un árbol",
        },
    )
    radio_answer(
        "a33_2",
        "¿Crees que el mobiliario es lo suficientemente cómodo para usar el lugar durante todas las estaciones del año?",
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
    if A0_3 == 1:
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
    if a41_1 == 1:
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
    if A0_3 == 1:
        radio_answer(
            "a44_1",
            "¿Ha aumentado el número de negocios o unidades económicas alrededor del lugar a raíz de la intervención?",
            options=[1, 2, 3],
            labels_map={1: "Sí, bastantes", 2: "Sí, por lo menos una", 3: "No, ninguna"},
        )
        radio_answer(
            "a44_2",
            "¿Has percibido un mayor ingreso en tu negocio a partir de la intervención en el lugar?",
            options=[1, 2, 3],
            labels_map={1: "Sí, muy directamente", 2: "Sí, pero no sé si es por la intervención", 3: "La verdad no"},
        )
    else:
        st.markdown("En esta evaluación no se recabó información directa de negocios o unidades económicas locales.")

        clear_branch("a44_1", "a44_2")
        set_ans("a44_1", None)
        set_ans("a44_2", None)

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

    # ========== A1 ==========
    a11_1_val = ans.get("a11_1")
    a11_1 = None if ans.get("a11_1_na") else (float(a11_1_val) if a11_1_val is not None else None)
    a11_2_val = ans.get("a11_2")
    a11_2 = None if ans.get("a11_2_na") else (float(a11_2_val) if a11_2_val is not None else None)

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

    A1_indicators = {
        "A1.1": A1_1,
        "A1.2": A1_2,
        "A1.3": A1_3,
        "A1.4": A1_4,
        "A1.5": A1_5,
    }

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

    A3_indicators = {
        "A3.1": A3_1,
        "A3.2": A3_2,
        "A3.3": A3_3,
        "A3.4": A3_4,
        "A3.5": A3_5,
        "A3.6": A3_6,
        "A3.7": A3_7,
    }

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
    a22_2 = ans.get("a22_2")
    a22_3 = ans.get("a22_3")
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

    A2_indicators = {
        "A2.1": A2_1,
        "A2.2": A2_2,
        "A2.3": A2_3,
        "A2.4": A2_4,
        "A2.5": A2_5,
        "A2.6": A2_6,
    }

    # ========== A4 ==========
    a41_1 = ans.get("a41_1")
    a41_before = ans.get("a41_before", 0)
    a41_after = ans.get("a41_after", 0)
    a41_2 = ans.get("a41_2")
    a42_1 = ans.get("a42_1")
    a43_1 = ans.get("a43_1")
    a43_2 = ans.get("a43_2")
    a44_1 = ans.get("a44_1")
    a44_2 = ans.get("a44_2")
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

    A4_indicators = {
        "A4.1": A4_1,
        "A4.2": A4_2,
        "A4.3": A4_3,
        "A4.4": A4_4,
        "A4.5": A4_5,
        "A4.6": A4_6,
    }

    # ===== Intangibles =====
    intangibles_A1 = calc_intangibles_A1(A1_1, A1_2, A1_3, A1_4, A1_5)
    intangibles_A2 = calc_intangibles_A2(A2_1, A2_2, A2_3, A2_4, A2_5, A2_6, A3_4)
    intangibles_A3 = calc_intangibles_A3(A3_1, A3_2, A3_3, A3_4, A3_5, A3_6, A3_7)
    intangibles_A4 = calc_intangibles_A4(A4_1, A4_2, A4_3, A4_4, A4_5, A4_6)

    # ===== Totales y global =====
    A1_total = compute_attribute_total(program_id, "A1", A1_indicators)
    A2_total = compute_attribute_total(program_id, "A2", A2_indicators)
    A3_total = compute_attribute_total(program_id, "A3", A3_indicators)
    A4_total = compute_attribute_total(program_id, "A4", A4_indicators)

    section_scores = compute_section_scores(A1_total, A2_total, A3_total, A4_total)
    global_score = compute_global_score(program_id, section_scores)

    # ===== Resumen numérico =====
    st.subheader("Resumen de atributos")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Encuentro", f"{A1_total:.1f}")
    c2.metric("Conexiones y Accesos", f"{A2_total:.1f}")
    c3.metric("Comodidad e Imagen", f"{A3_total:.1f}")
    c4.metric("Usos y Actividades", f"{A4_total:.1f}")
    c5.metric("Resultado global", f"{global_score:.1f}")

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
                "Comunitario",
            ],
            "outer_labels": ["A4.1", "A4.2", "A4.3", "A4.4", "A4.5", "A4.6"],
        },
    }

    scores_middle = {
        "Usos y Actividades": intangibles_A4,
        "Comodidad e Imagen": intangibles_A3,
        "Conexiones y Accesos": intangibles_A2,
        "Encuentro": intangibles_A1,
    }
    scores_outer = {
        "Usos y Actividades": A4_indicators,
        "Comodidad e Imagen": A3_indicators,
        "Conexiones y Accesos": A2_indicators,
        "Encuentro": A1_indicators,
    }

    # ──────────────────────────────────────────
    # Rueda global  (drop-in replacement)
    # ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("<h3 style='text-align: center;'>Diagrama de Lugar</h3>", unsafe_allow_html=True)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.textpath import TextPath
    from matplotlib.patches import PathPatch
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Affine2D
    from pycirclize import Circos

    # Use Roboto everywhere (make sure the font is available in your system)
    plt.rcParams["font.family"] = "Roboto"

    # ----- 1.  COLORS --------------------------------------------------------------
    track_colors = {
        "Usos y Actividades": {"inner": "#D15E4C", "middle": "#E09F92", "outer": "#F4D8CE"},
        "Comodidad e Imagen": {"inner": "#A8BA4F", "middle": "#CDD492", "outer": "#EEF2CB"},
        "Conexiones y Accesos": {"inner": "#25447E", "middle": "#6F8EB4", "outer": "#B2C3D3"},
        "Encuentro":           {"inner": "#575693", "middle": "#9D9EBA", "outer": "#D0D3D9"},
    }

    # ----- 2.  RADIO ---------------------------------------------------------------
    r_center_min, r_center_max = 0,   5          # disco central
    r_inner_min,  r_inner_max  = 10, 45          # anillo del atributo (sector title vive aquí)
    r_mid_min,    r_mid_max    = 50, 85          # anillo intangibles
    r_outer_min,  r_outer_max  = 80, 115         # anillo indicadores

    # ----- 3.  SECTORES Y PUNTAJES -------------------------------------------------
    sectors = {name: 90 for name in sector_names}           # 4 × 90 °
    circos = Circos(sectors, space=0)                       # sin huecos

    #  «sector_scores» ya existe en tu código; si no, constrúyelo igual que antes
    # (Este bloque es idéntico al tuyo; lo incluyo por claridad)
    sector_scores = {}
    for name in sector_names:
        mid_labels  = SECTOR_CONFIG[name]["middle_labels"]
        out_labels  = SECTOR_CONFIG[name]["outer_labels"]
        middle_pcts = [scores_middle.get(name,  {}).get(lbl, 0.0) for lbl in mid_labels]
        outer_pcts  = [scores_outer .get(name,  {}).get(lbl, 0.0) for lbl in out_labels]
        sector_scores[name] = {
            "middle_pcts":  middle_pcts,
            "outer_pcts":   outer_pcts,
            "middle_labels": mid_labels,
            "outer_labels":  out_labels,
        }

    # ----- 5.  FIGURA BASE ---------------------------------------------------------
    fig = circos.plotfig(figsize=(4.5, 4.5), dpi=90)
    ax  = fig.axes[0]

    # ----- 4.  DIBUJO DE LOS ANILLOS ----------------------------------------------
    for sector in circos.sectors:
        name   = sector.name
        colors = track_colors[name]
        length = sector.size

        middle_pcts = sector_scores[name]["middle_pcts"]
        outer_pcts  = sector_scores[name]["outer_pcts"]

        # ANILLO INNER (fondo del atributo)
        t_inner = sector.add_track((r_inner_min, r_inner_max))
        t_inner.rect(0, length, fc=colors["inner"], ec="white", lw=4)

        # ANILLO MIDDLE (intangibles)
        if middle_pcts:
            blk = length / len(middle_pcts)
            for i, pct in enumerate(middle_pcts):
                start, end = i * blk, (i + 1) * blk
                ratio = nz(pct) / 100.0
                # Color lleno
                if ratio > 0:
                    t = sector.add_track((r_mid_min,
                                        r_mid_min + (r_mid_max - r_mid_min) * ratio))
                    t.rect(start, end, fc=colors["middle"], ec="white", lw=4)
                # Color vacío
                if ratio < 1:
                    t = sector.add_track((r_mid_min + (r_mid_max - r_mid_min) * ratio,
                                        r_mid_max))
                    t.rect(start, end, fc="#FFFFFF", ec="white", lw=4)

        # ANILLO OUTER (indicadores)
        if outer_pcts:
            blk = length / len(outer_pcts)
            for i, pct in enumerate(outer_pcts):
                start, end = i * blk, (i + 1) * blk
                ratio = nz(pct) / 100.0
                # Lleno
                if ratio > 0:
                    t = sector.add_track((r_outer_min,
                                        r_outer_min + (r_outer_max - r_outer_min) * ratio))
                    t.rect(start, end, fc=colors["outer"], ec="white", lw=6)
                # Vacío
                if ratio < 1:
                    t = sector.add_track((r_outer_min + (r_outer_max - r_outer_min) * ratio,
                                        r_outer_max))
                    t.rect(start, end, fc="#FFFFFF", ec="white", lw=6)

    # ----- 5.  FIGURA BASE ---------------------------------------------------------
    fig = circos.plotfig(figsize=(4.5, 4.5), dpi=90)
    ax  = fig.axes[0]


    # ----- 8.  DISCO CENTRAL Y TEXTO “LUGAR” ---------------------------------------




    # ----- 7.  MOSTRAR EN STREAMLIT ------------------------------------------------
    st.pyplot(fig, use_container_width=True)

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
            "Comunitario": 5,
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
    r_inner_min_q, r_inner_max_q = 0.00, 0.20
    r_mid_min_q, r_mid_max_q = 0.20, 0.55
    r_outer_min_q, r_outer_max_q = 0.55, 0.95

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
            "Valor (0–100)": [f"{nz(v):.1f}" for v in ind_vals],
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
            "Valor (0–100)": [f"{nz(v):.1f}" for v in intangible_vals],
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
        "A1.1": A1_1,
        "A1.2": A1_2,
        "A1.3": A1_3,
        "A1.4": A1_4,
        "A1.5": A1_5,
        "A1_total": A1_total,
        "A2.1": A2_1,
        "A2.2": A2_2,
        "A2.3": A2_3,
        "A2.4": A2_4,
        "A2.5": A2_5,
        "A2.6": A2_6,
        "A2_total": A2_total,
        "A3.1": A3_1,
        "A3.2": A3_2,
        "A3.3": A3_3,
        "A3.4": A3_4,
        "A3.5": A3_5,
        "A3.6": A3_6,
        "A3.7": A3_7,
        "A3_total": A3_total,
        "A4.1": A4_1,
        "A4.2": A4_2,
        "A4.3": A4_3,
        "A4.4": A4_4,
        "A4.5": A4_5,
        "A4.6": A4_6,
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
        st.markdown(
            "<div style='margin-top: 12px; text-align:center;'>"
            "<a href='#tab_selector' class='tab-selector-link'>"
            "Volver al selector de pestañas</a></div>",
            unsafe_allow_html=True,
        )

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
    st.markdown(
        "<div style='margin-top: 12px; text-align:center;'>"
        "<a href='#tab_selector' class='tab-selector-link'>"
        "Volver al selector de pestañas</a></div>",
        unsafe_allow_html=True,
    )
