try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import asyncio
import base64
import re
import os
import unicodedata
from datetime import datetime
from typing import Optional
import cv2
import easyocr
import httpx
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Header
from pdf2image import convert_from_bytes
from pyzbar import pyzbar
from groq_service import generar_resumen_groq
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from ine_ocr_v2 import extraer_ine_frente_v2, extraer_ine_reverso_v2, combinar_ine_v2

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY          = os.getenv("GROQ_API_KEY", "")
N8N_WEBHOOK_URL         = os.getenv("N8N_WEBHOOK_URL", "")
DIAS_ALERTA_VENCIMIENTO = int(os.getenv("DIAS_ALERTA_VENCIMIENTO", "30"))

SUGERENCIAS_DOC = {
    "FORMATO_CURP": [
        "Verifica que el CURP coincida con otros documentos; inconsistencias pueden invalidar trámites.",
        "Se recomienda usar una CURP descargada recientemente desde el portal oficial para asegurar "
        "que esté certificada y actualizada. Algunas instituciones pueden rechazar versiones antiguas.",
    ],
    "PASAPORTE": [
        "Pasaporte próximo a vencer puede no ser aceptado en trámites o viajes.",
    ],
    "CONSTANCIA_SAT": [
        "Valida que los datos coincidan exactamente con el SAT. Errores mínimos pueden impedir facturación.",
    ],
    "INE": [
        "Es el principal medio de identificación oficial.",
        "No compartas fotos completas públicamente.",
        "Documento vencido o ilegible puede invalidar tu identificación oficial.",
    ],
    "ACTA_NACIMIENTO": [
        "Guarda el acta en buen estado; es un documento base para muchos trámites.",
    ],
    "GENERAL": [
        "No compartir documentos por WhatsApp, correo o terceros sin verificar.",
        "Mantener documentos en buen estado (físico y digital).",
    ],
}

# ─── Regex globales ───────────────────────────────────────────────────────────
CURP_REGEX        = r'[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]{2}'
CURP_REGEX_STRICT = r'^[A-Z]{4}\d{6}[HM][A-Z]{2}[BCDFGHJKLMNPQRSTVWXYZ]{3}[A-Z0-9]\d$'

# ─── Supabase ─────────────────────────────────────────────────────────────────
try:
    from supabase_service import (
        guardar_resultado_documento,
        obtener_documentos_usuario,
        obtener_documento,
        descargar_archivo_storage,
        obtener_estadisticas,
        marcar_revisado,
        obtener_rol_usuario,
        verificar_token_supabase,
    )
    SUPABASE_DISPONIBLE = True
except Exception as e:
    print(f"⚠ Supabase no configurado: {e}")
    SUPABASE_DISPONIBLE = False

app = FastAPI(title="SmartDocs OCR", version="3.5")

print("⚡ Cargando EasyOCR...")
reader = easyocr.Reader(['es', 'en'])
print("✅ Motor OCR Listo.")
ocr_semaphore = asyncio.Semaphore(1) 

OcrResult = list[tuple]


# ===========================================================================
# PREPROCESAMIENTO
# ===========================================================================

def detectar_angulo_rotacion(imagen: np.ndarray) -> float:
    edges = cv2.Canny(imagen, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return 0.0
    angles = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        if abs(angle) < 45:
            angles.append(angle)
    return float(np.median(angles)) if angles else 0.0


def corregir_rotacion(imagen: np.ndarray) -> np.ndarray:
    angulo = detectar_angulo_rotacion(imagen)
    if abs(angulo) < 0.5:
        return imagen
    (h, w) = imagen.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angulo, 1.0)
    return cv2.warpAffine(imagen, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def mejorar_contraste(imagen: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(imagen)


def binarizar(imagen: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        imagen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)


def evaluar_calidad(imagen: np.ndarray) -> dict:
    nitidez = float(cv2.Laplacian(imagen, cv2.CV_64F).var())
    brillo  = float(np.mean(imagen))
    advertencias = []
    if nitidez < 100:
        advertencias.append("Imagen posiblemente borrosa — intenta tomar la foto en mejor luz")
    if brillo < 50:
        advertencias.append("Imagen muy oscura")
    if brillo > 245:
        advertencias.append("Imagen sobreexpuesta")
    return {
        "nitidez_score": round(nitidez, 2),
        "brillo_promedio": round(brillo, 2),
        "advertencias": advertencias,
        "calidad_suficiente": len(advertencias) == 0,
    }


def bytes_a_gris(file_bytes: bytes, filename: str) -> np.ndarray:
    if filename.lower().endswith('.pdf'):
        images = convert_from_bytes(file_bytes)
        if not images:
            raise ValueError("PDF vacío o corrupto")
        return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2GRAY)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Archivo de imagen corrupto o formato no soportado")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def bytes_a_bgr(file_bytes: bytes, filename: str) -> np.ndarray:
    if filename.lower().endswith('.pdf'):
        images = convert_from_bytes(file_bytes)
        if not images:
            raise ValueError("PDF vacío")
        return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagen corrupta")
    return img


def preprocesar(file_bytes: bytes, filename: str) -> tuple[np.ndarray, dict]:
    img_gris = bytes_a_gris(file_bytes, filename)
    calidad  = evaluar_calidad(img_gris)
    img = mejorar_contraste(img_gris)
    img = corregir_rotacion(img)
    if calidad['brillo_promedio'] <= 200:
        img = binarizar(img)
    return img, calidad

def preprocesar_ine(file_bytes: bytes, filename: str) -> tuple[np.ndarray, dict]:
    """
    Pipeline de preprocesamiento optimizado para INE.
    Incluye escalado y sharpen que son críticos para credenciales.
    """
    img_gris = bytes_a_gris(file_bytes, filename)
    calidad  = evaluar_calidad(img_gris)

    # 1. Escalar (CRÍTICO para credenciales pequeñas)
    h, w = img_gris.shape
    if max(h, w) < 1500:
        scale = 2
        img_gris = cv2.resize(img_gris, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)

    # 2. Reducir ruido preservando bordes
    img = cv2.bilateralFilter(img_gris, 9, 75, 75)

    # 3. Mejorar contraste
    img = mejorar_contraste(img)

    # 4. Sharpen (CRÍTICO para texto pequeño de la INE)
    kernel = np.array([[0, -1, 0],
                       [-1,  5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # 5. Corregir rotación
    img = corregir_rotacion(img)

    # 6. Binarización solo si es necesario
    if calidad['brillo_promedio'] <= 200:
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 5
        )

    # 7. Limpieza morfológica
    kernel_m = np.ones((2, 2), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_m)

    return img, calidad


def limpiar_ocr(resultado_ocr: OcrResult) -> OcrResult:
    """Filtra bloques OCR con baja confianza o texto muy corto."""
    return [
        (bbox, texto, conf)
        for (bbox, texto, conf) in resultado_ocr
        if conf > 0.2 and len(texto.strip()) > 1
    ]

def leer_codigos(file_bytes: bytes, filename: str) -> list[dict]:
    img_bgr = bytes_a_bgr(file_bytes, filename)
    codigos = pyzbar.decode(img_bgr)
    resultados = []
    for c in codigos:
        try:
            data = c.data.decode('utf-8')
        except UnicodeDecodeError:
            data = c.data.decode('latin-1', errors='ignore')
        resultados.append({"tipo": c.type, "datos": data})
    return resultados


# ===========================================================================
# QR — INE
# ===========================================================================

def _es_qr_url_ine(datos: str) -> bool:
    return datos.strip().startswith("http") and "qr.ine.mx" in datos


async def _fetch_datos_qr_ine_url(url: str) -> dict:
    datos = {}
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; SmartDocs/3.5)"
            })
            resp.raise_for_status()
            html = resp.text.upper()

            patrones_campo = {
                'apellido_paterno': [
                    r'APELLIDO\s+PATERNO[:\s<>/\w]*?([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{1,40}?)(?:<|\n|,|\|)',
                    r'PRIMER\s+APELLIDO[:\s<>/\w]*?([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{1,40}?)(?:<|\n|,|\|)',
                ],
                'apellido_materno': [
                    r'APELLIDO\s+MATERNO[:\s<>/\w]*?([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{1,40}?)(?:<|\n|,|\|)',
                    r'SEGUNDO\s+APELLIDO[:\s<>/\w]*?([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{1,40}?)(?:<|\n|,|\|)',
                ],
                'nombre': [
                    r'NOMBRE\(?S?\)?[:\s<>/\w]*?([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{1,40}?)(?:<|\n|,|\|)',
                ],
                'curp': [
                    r'CURP[:\s<>/\w]*?([A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]{2})',
                ],
                'fecha_nacimiento': [
                    r'FECHA\s+(?:DE\s+)?NACIMIENTO[:\s<>/\w]*?(\d{2}/\d{2}/\d{4})',
                    r'FECHA\s+(?:DE\s+)?NACIMIENTO[:\s<>/\w]*?(\d{4}-\d{2}-\d{2})',
                ],
                'sexo': [
                    r'SEXO[:\s<>/\w]*?(HOMBRE|MUJER|MASCULINO|FEMENINO|[HMF])\b',
                ],
                'clave_elector': [
                    r'CLAVE\s+(?:DE\s+)?ELECTOR[:\s<>/\w]*?([A-Z]{6}\d{8}[A-Z]\d{3})',
                ],
                'vigencia': [
                    r'VIGENCIA[:\s<>/\w]*?(\d{4})',
                ],
            }

            for campo_key, patrones in patrones_campo.items():
                for pat in patrones:
                    m = re.search(pat, html)
                    if m:
                        datos[campo_key] = m.group(1).strip()
                        break

            for k, v in list(datos.items()):
                if isinstance(v, str):
                    datos[k] = re.sub(r'<[^>]+>', '', v).strip()

            print(f"✅ QR INE URL consultado — campos obtenidos: {list(datos.keys())}")

    except httpx.HTTPStatusError as e:
        print(f"⚠ QR INE URL error HTTP {e.response.status_code}: {url}")
    except Exception as e:
        print(f"⚠ QR INE URL no accesible: {e}")

    return datos


def parsear_qr_ine_pipe(datos_qr: str) -> dict:
    partes = [p.strip() for p in datos_qr.split('|')]
    campos = [
        'id_registro', 'apellido_paterno', 'apellido_materno', 'nombre',
        'nombre_completo', 'sexo', 'fecha_nacimiento', 'curp', 'clave_elector',
        'numero_emision', 'numero_ocr', 'vigencia',
    ]
    return {campos[i]: partes[i]
            for i in range(min(len(campos), len(partes))) if partes[i]}


async def obtener_datos_qr_ine(codigos: list[dict]) -> tuple[dict, str]:
    for cod in codigos:
        if cod.get('tipo') != 'QRCODE':
            continue
        datos_str = cod.get('datos', '')
        if _es_qr_url_ine(datos_str):
            datos = await _fetch_datos_qr_ine_url(datos_str)
            return datos, 'URL_2023'
        if '|' in datos_str:
            datos = parsear_qr_ine_pipe(datos_str)
            return datos, 'PIPE_2022'
    return {}, 'NINGUNO'


# ===========================================================================
# QR / CODE128 — CURP y Acta
# ===========================================================================

def _es_curp_valida(texto: str) -> bool:
    """Verifica si un string es una CURP válida (estructura + fecha)."""
    texto = texto.strip().upper()
    if len(texto) != 18:
        return False
    if not re.match(CURP_REGEX_STRICT, texto):
        return False
    try:
        datetime.strptime(texto[4:10], '%y%m%d')
    except ValueError:
        return False
    return True


def clasificar_qr_curp_o_acta(datos_str: str) -> tuple[str, dict]:
    """
    Analiza un QR pipe-separated y determina si es un documento CURP
    (primer campo = CURP válida) o un acta de nacimiento.
    """
    partes = [p.strip() for p in datos_str.split('|')]
    primer_campo = partes[0] if partes else ''

    # ── Caso 1: primer campo es CURP válida → documento CURP ─────────────────
    if _es_curp_valida(primer_campo):
        datos: dict = {'curp': primer_campo}
        idx = 1
        while idx < len(partes) and not partes[idx]:
            idx += 1
        campos_esperados = [
            'apellido_paterno', 'apellido_materno', 'nombre',
            'sexo', 'fecha_nacimiento', 'entidad', 'municipio',
        ]
        for k in campos_esperados:
            if idx < len(partes) and partes[idx]:
                datos[k] = partes[idx]
            idx += 1
        return 'FORMATO_CURP', datos

    # ── Caso 2: primer campo NO es CURP → acta pipe-separated ────────────────
    # Usar mapeo por índice para no confundir campos
    datos_acta: dict = {}
    campos_acta_pipe = [
        'version', 'anio_registro', 'clave_entidad', '_tipo_acta',
        'libro', '_num_interno', '_flag1', '_vacio1',
        '_vacio2', 'dia_nacimiento_raw', '_vacio3', 'clave_registro_civil',
        'fecha_registro', '_estado_acta', 'nombre', 'apellido_paterno',
        'apellido_materno', 'sexo_raw', 'fecha_nacimiento',
    ]
    for i, k in enumerate(campos_acta_pipe):
        if i < len(partes) and partes[i].strip() and not k.startswith('_'):
            datos_acta[k] = partes[i].strip()

    # Campos extendidos: CURP en índice 23, padres en 24-26 / 30-32
    _idx_ext = {23: 'curp', 24: 'nombre_padre', 25: 'apellido_paterno_padre',
                26: 'apellido_materno_padre', 30: 'nombre_madre',
                31: 'apellido_paterno_madre', 32: 'apellido_materno_madre'}
    for idx, k in _idx_ext.items():
        if idx < len(partes) and partes[idx].strip():
            datos_acta[k] = partes[idx].strip()

    # Post-procesar: normalizar sexo
    if datos_acta.get('sexo_raw'):
        sv = datos_acta.pop('sexo_raw').upper()
        datos_acta['sexo'] = 'H' if sv in ('H', 'M', 'MASCULINO', 'HOMBRE') else 'M'

    # Consolidar nombre completo padres
    if all(datos_acta.get(k) for k in ('nombre_padre', 'apellido_paterno_padre', 'apellido_materno_padre')):
        datos_acta['padre'] = (f"{datos_acta.pop('nombre_padre')} "
                               f"{datos_acta.pop('apellido_paterno_padre')} "
                               f"{datos_acta.pop('apellido_materno_padre')}")
    if all(datos_acta.get(k) for k in ('nombre_madre', 'apellido_paterno_madre', 'apellido_materno_madre')):
        datos_acta['madre'] = (f"{datos_acta.pop('nombre_madre')} "
                               f"{datos_acta.pop('apellido_paterno_madre')} "
                               f"{datos_acta.pop('apellido_materno_madre')}")

    return 'ACTA_NACIMIENTO', datos_acta


def parsear_code128_acta(datos_codigo: str) -> dict:
    """
    Parsea un CODE128.
    Si el valor completo es una CURP válida, lo marca como curp_code128.
    """
    datos = {}
    partes = datos_codigo.strip().split()

    if len(partes) == 1:
        if _es_curp_valida(partes[0]):
            datos['curp_code128'] = partes[0]
        else:
            datos['numero_acta_codigo'] = partes[0]
        return datos

    if len(partes) == 2:
        datos['clave_renapo']       = partes[0]
        datos['numero_acta_codigo'] = partes[1]

    return datos


def _extraer_payload_ascii_qr_acta(raw: bytes) -> str | None:
    candidatos = []
    for m in re.finditer(rb'[\x20-\x7E]{20,}', raw):
        txt = m.group(0).decode('latin-1', errors='ignore')
        if txt.count('|') >= 5:
            candidatos.append(txt)
    if not candidatos:
        return None
    return max(candidatos, key=lambda t: (t.count('|'), len(t)))


# ── Mapeo por índice para el formato RENAPO/estados de actas digitales ────────
# Formato real observado (35+ campos pipe-separated dentro del CMS/PKCS7):
#  0:version | 1:anio_reg | 2:entidad | 3:tipo | 4:libro | 5:num_int |
#  6:flag | 7-8:vacíos | 9:dia_nac | 10:vacio | 11:clave_rc |
#  12:fecha_reg | 13:estado_acta | 14:nombre | 15:ap_pat | 16:ap_mat |
#  17:sexo | 18:fecha_nac | 19-21:meta | 22:mun_cod | 23:curp |
#  24:nombre_padre | 25:ap_pat_padre | 26:ap_mat_padre | ...
#  30:nombre_madre | 31:ap_pat_madre | 32:ap_mat_madre | ...
_IDX_CAMPOS_QR_ACTA = {
    0:  'version',
    1:  'anio_registro',
    2:  'clave_entidad',
    4:  'libro',
    9:  '_dia_nacimiento_raw',
    11: 'clave_registro_civil',
    12: 'fecha_registro',
    14: 'nombre',
    15: 'apellido_paterno',
    16: 'apellido_materno',
    17: '_sexo_raw',
    18: 'fecha_nacimiento',
    22: 'municipio_codigo',
    23: 'curp',
    24: '_nombre_padre',
    25: '_ap_pat_padre',
    26: '_ap_mat_padre',
    30: '_nombre_madre',
    31: '_ap_pat_madre',
    32: '_ap_mat_madre',
}


def parsear_qr_acta_base64(datos_qr: str) -> dict:
    """
    Decodifica el QR del acta (base64 → PKCS7/CMS → pipe-separated).
    Usa mapeo por índice fijo para el formato real RENAPO.
    No confunde con documentos CURP (guard por primer campo).
    """
    # Guard: si ya es pipe-separated y primer campo es CURP → no es acta
    if '|' in datos_qr:
        partes = datos_qr.split('|')
        if partes and _es_curp_valida(partes[0].strip()):
            return {}

    try:
        raw = base64.b64decode(datos_qr)
    except Exception:
        return {}

    # Intentar decodificar texto del payload PKCS7/CMS
    texto = None
    for encoding in ('utf-8', 'latin-1'):
        try:
            candidato = raw.decode(encoding, errors='ignore')
            if candidato.count('|') >= 5:
                texto = candidato
                break
        except Exception:
            continue

    # Fallback: buscar segmento ASCII con pipes dentro del binario
    if not texto:
        payload = _extraer_payload_ascii_qr_acta(raw)
        if not payload:
            return {}
        texto = payload

    partes = [p.strip() for p in texto.split('|')]

    # Guard: primer campo CURP → es doc CURP, no acta
    if partes and _es_curp_valida(partes[0]):
        return {}

    # ── Mapeo por índice ──────────────────────────────────────────────────────
    resultado: dict = {}
    for idx, key in _IDX_CAMPOS_QR_ACTA.items():
        if idx < len(partes) and partes[idx].strip():
            resultado[key] = partes[idx].strip()

    # ── Consolidar nombre completo del padre ──────────────────────────────────
    if all(resultado.get(k) for k in ('_nombre_padre', '_ap_pat_padre', '_ap_mat_padre')):
        resultado['padre'] = (f"{resultado.pop('_nombre_padre')} "
                              f"{resultado.pop('_ap_pat_padre')} "
                              f"{resultado.pop('_ap_mat_padre')}")
    else:
        for k in ('_nombre_padre', '_ap_pat_padre', '_ap_mat_padre'):
            resultado.pop(k, None)

    # ── Consolidar nombre completo de la madre ────────────────────────────────
    if all(resultado.get(k) for k in ('_nombre_madre', '_ap_pat_madre', '_ap_mat_madre')):
        resultado['madre'] = (f"{resultado.pop('_nombre_madre')} "
                              f"{resultado.pop('_ap_pat_madre')} "
                              f"{resultado.pop('_ap_mat_madre')}")
    else:
        for k in ('_nombre_madre', '_ap_pat_madre', '_ap_mat_madre'):
            resultado.pop(k, None)

    # ── Normalizar sexo (F=Femenino→M, H=Hombre→H) ───────────────────────────
    if resultado.get('_sexo_raw'):
        sv = resultado.pop('_sexo_raw').upper()
        resultado['sexo'] = 'H' if sv in ('H', 'MASCULINO', 'HOMBRE') else 'M'

    # ── Normalizar fechas a ISO ───────────────────────────────────────────────
    for fkey in ('fecha_nacimiento', 'fecha_registro'):
        if resultado.get(fkey):
            iso = normalizar_fecha(resultado[fkey])
            if iso:
                resultado[fkey] = iso

    # ── Limpiar campos internos residuales ────────────────────────────────────
    resultado.pop('_dia_nacimiento_raw', None)

    return resultado


# ===========================================================================
# VALIDACIONES
# ===========================================================================

def validar_curp(curp: str) -> bool:
    return _es_curp_valida(curp)


def validar_rfc(rfc: str) -> bool:
    if not rfc:
        return False
    return bool(re.match(r'^[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3}$', rfc))


def normalizar_fecha(texto: str) -> str | None:
    for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(texto, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


def calcular_dias_vencimiento(fecha_iso: str) -> int | None:
    try:
        return (datetime.strptime(fecha_iso, '%Y-%m-%d') - datetime.now()).days
    except Exception:
        return None


_MESES_ES_MX = {
    'ENERO': '01', 'FEBRERO': '02', 'MARZO': '03', 'ABRIL': '04',
    'MAYO': '05', 'JUNIO': '06', 'JULIO': '07', 'AGOSTO': '08',
    'SEPTIEMBRE': '09', 'OCTUBRE': '10', 'NOVIEMBRE': '11', 'DICIEMBRE': '12',
}


def normalizar_fecha_es_mx(texto: str) -> str | None:
    texto = texto.upper().strip().replace('.', '')
    texto = re.sub(r'\s+', ' ', texto)
    for fmt in ('%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(texto, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    m = re.search(r'(\d{1,2})\s+DE\s+([A-ZÁÉÍÓÚÑ]+)\s+DE\s+(\d{4})', texto)
    if m:
        dia, mes_txt, anio = m.group(1), m.group(2), m.group(3)
        mes = _MESES_ES_MX.get(mes_txt)
        if mes:
            return f"{anio}-{mes}-{dia.zfill(2)}"
    return None


# ===========================================================================
# HELPERS OCR
# ===========================================================================

def campo(valor, confianza: float, valido: bool = None, fuente: str = "ocr") -> dict:
    r = {"valor": valor, "confianza": round(confianza, 3), "fuente": fuente}
    if valido is not None:
        r["valido"] = valido
    return r


def buscar_con_confianza(resultado_ocr: OcrResult, patron: str) -> tuple[str | None, float]:
    texto_acum, mapa = "", []
    for (bbox, texto, conf) in resultado_ocr:
        ini = len(texto_acum)
        texto_acum += texto.upper() + " "
        mapa.append((ini, len(texto_acum), conf))
    match = re.search(patron, texto_acum)
    if not match:
        return None, 0.0
    confs = [c for (ini, fin, c) in mapa if ini <= match.end() and fin >= match.start()]
    return match.group(0), float(np.mean(confs)) if confs else 0.0


def texto_plano(resultado_ocr: OcrResult) -> str:
    return " ".join(t.upper() for (_, t, _) in resultado_ocr)


def _conf_bloque(resultado_ocr: OcrResult, valor: str) -> float:
    valor = valor.upper()
    confs = [c for (_, t, c) in resultado_ocr if valor in t.upper()]
    return float(np.mean(confs)) if confs else 0.7


def _lineas_ordenadas(resultado_ocr: OcrResult) -> list[tuple[float, str, float]]:
    lineas = []
    for (bbox, texto, conf) in resultado_ocr:
        ys = [p[1] for p in bbox]
        lineas.append((float(np.mean(ys)), texto.upper().strip(), conf))
    lineas.sort(key=lambda x: x[0])
    return lineas


# ===========================================================================
# HELPER — normalización de texto para scoring
# ===========================================================================

def _quitar_acentos(texto: str) -> str:
    """Elimina diacríticos para comparaciones robustas en scoring."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )


# ===========================================================================
# QR → formato campo()
# ===========================================================================

def datos_qr_ine_a_campos(raw: dict, modelo: str = 'PIPE_2022') -> dict:
    fuente = f"QR_{modelo}"
    datos  = {}

    def _c(val, **kw):
        return campo(val, 0.99, fuente=fuente, **kw)

    if raw.get('curp'):
        v = raw['curp']
        datos['curp'] = campo(v, 0.99, validar_curp(v), fuente=fuente)

    for key in ('apellido_paterno', 'apellido_materno', 'nombre',
                'nombre_completo', 'sexo', 'fecha_nacimiento',
                'clave_elector', 'numero_ocr', 'numero_emision'):
        if raw.get(key):
            datos[key] = _c(raw[key])

    if 'nombre_completo' not in datos:
        partes = [raw.get(k, '') for k in ('apellido_paterno', 'apellido_materno', 'nombre')]
        if all(partes):
            datos['nombre_completo'] = _c(' '.join(partes))

    if raw.get('vigencia'):
        year       = raw['vigencia']
        fecha_venc = f"{year}-12-31"
        dias       = calcular_dias_vencimiento(fecha_venc)
        datos['vigencia']          = _c(year)
        datos['fecha_vencimiento'] = _c(fecha_venc)
        datos['dias_para_vencer']  = dias
        datos['vencido']           = dias is not None and dias < 0

    return datos


def datos_qr_curp_a_campos(raw: dict) -> dict:
    """Convierte el dict del QR de un documento CURP al formato campo()."""
    datos = {}

    if raw.get('curp'):
        cv = raw['curp']
        datos['curp'] = campo(cv, 0.99, validar_curp(cv), fuente='QR_CURP')

    for key in ('apellido_paterno', 'apellido_materno', 'nombre',
                'fecha_nacimiento', 'entidad', 'municipio'):
        if raw.get(key):
            datos[key] = campo(raw[key], 0.99, fuente='QR_CURP')

    if raw.get('sexo'):
        raw_sexo = raw['sexo'].upper()
        curp     = raw.get('curp', '')
        if raw_sexo == 'MUJER':
            sexo_norm = 'F'
        elif raw_sexo == 'HOMBRE':
            sexo_norm = 'H'
        elif raw_sexo == 'H':
            sexo_norm = 'H'
        elif raw_sexo == 'M':
            sexo_norm = 'H' if (len(curp) >= 11 and curp[10] == 'H') else 'F'
        else:
            sexo_norm = raw_sexo
        datos['sexo'] = campo(sexo_norm, 0.99, fuente='QR_CURP')

    partes_nc = [raw.get(k, '') for k in ('nombre', 'apellido_paterno', 'apellido_materno')]
    if all(partes_nc):
        datos['nombre_completo'] = campo(' '.join(partes_nc), 0.99, fuente='QR_CURP')

    return datos


def combinar_datos_qr_ocr(datos_ocr: dict, datos_qr: dict) -> dict:
    """QR sobreescribe OCR campo por campo."""
    return {**datos_ocr, **datos_qr}


# ===========================================================================
# EXTRACCIÓN — INE frente
# ===========================================================================

_STOPWORDS_INE = {
    'INSTITUTO', 'NACIONAL', 'ELECTORAL', 'CREDENCIAL', 'PARA', 'VOTAR',
    'NOMBRE', 'APELLIDO', 'CURP', 'CLAVE', 'ELECTOR', 'VIGENCIA',
    'DOMICILIO', 'MUNICIPIO', 'ESTADO', 'SECCION', 'LOCALIDAD',
    'DISTRITO', 'FECHA', 'NACIMIENTO', 'REGISTRO', 'MEXICO', 'EDAD',
    'SEXO', 'FIRMA', 'ANO', 'EMISION', 'FOLIO', 'INE', 'IFE',
}
_ETIQUETAS_AP_PAT  = ['APELLIDO PATERNO', 'PRIMER APELLIDO', 'AP PATERNO', 'AP. PATERNO']
_ETIQUETAS_AP_MAT  = ['APELLIDO MATERNO', 'SEGUNDO APELLIDO', 'AP MATERNO', 'AP. MATERNO']
_ETIQUETAS_NOMBRE  = ['NOMBRE(S)', 'NOMBRE(S):', 'NOMBRE:', 'NOMBRES', 'NOMBRE']
_ETIQUETAS_FEC_NAC = ['FECHA DE NACIMIENTO', 'F. NACIMIENTO', 'F.NACIMIENTO', 'FECHA NAC', 'NACIMIENTO']


def _es_nombre_valido(texto: str) -> bool:
    palabras = texto.strip().split()
    return (1 <= len(palabras) <= 4
            and all(re.match(r'^[A-ZÁÉÍÓÚÑ]+$', p) for p in palabras)
            and not any(p in _STOPWORDS_INE for p in palabras))


def _extraer_nombre_por_posicion(resultado_ocr: OcrResult) -> dict:
    datos  = {}
    lineas = _lineas_ordenadas(resultado_ocr)
    i = 0
    while i < len(lineas):
        _, txt, conf = lineas[i]

        if any(e in txt for e in _ETIQUETAS_AP_PAT) and 'apellido_paterno' not in datos:
            if i + 1 < len(lineas):
                _, sig, cs = lineas[i + 1]
                if _es_nombre_valido(sig):
                    datos['apellido_paterno'] = campo(sig.strip(), round(cs, 3))
                    i += 2; continue

        if any(e in txt for e in _ETIQUETAS_AP_MAT) and 'apellido_materno' not in datos:
            if i + 1 < len(lineas):
                _, sig, cs = lineas[i + 1]
                if _es_nombre_valido(sig):
                    datos['apellido_materno'] = campo(sig.strip(), round(cs, 3))
                    i += 2; continue

        if any(e in txt for e in _ETIQUETAS_NOMBRE) and 'nombre' not in datos:
            if i + 1 < len(lineas):
                _, sig, cs = lineas[i + 1]
                if _es_nombre_valido(sig):
                    datos['nombre'] = campo(sig.strip(), round(cs, 3))
                    i += 2; continue

        if any(e in txt for e in _ETIQUETAS_FEC_NAC) and 'fecha_nacimiento' not in datos:
            fm = re.search(r'(\d{2}/\d{2}/\d{4})', txt)
            if fm and normalizar_fecha(fm.group(1)):
                datos['fecha_nacimiento'] = campo(fm.group(1), round(conf, 3))
            elif i + 1 < len(lineas):
                _, sig, cs = lineas[i + 1]
                fm2 = re.search(r'(\d{2}/\d{2}/\d{4})', sig)
                if fm2 and normalizar_fecha(fm2.group(1)):
                    datos['fecha_nacimiento'] = campo(fm2.group(1), round(cs, 3))
                    i += 2; continue
        i += 1
    return datos


def extraer_ine_frente(resultado_ocr: OcrResult) -> dict:
    datos = {}
    texto = texto_plano(resultado_ocr)

    datos.update(_extraer_nombre_por_posicion(resultado_ocr))

    if 'apellido_paterno' not in datos:
        m = re.search(r'APELLIDO\s+PATERNO\s*:?\s*([A-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ]+){0,2})', texto)
        if m:
            datos['apellido_paterno'] = campo(m.group(1).strip(), _conf_bloque(resultado_ocr, m.group(1)))

    if 'apellido_materno' not in datos:
        m = re.search(r'APELLIDO\s+MATERNO\s*:?\s*([A-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ]+){0,2})', texto)
        if m:
            datos['apellido_materno'] = campo(m.group(1).strip(), _conf_bloque(resultado_ocr, m.group(1)))

    if 'nombre' not in datos:
        for pat in [
            r'NOMBRE\(?S?\)?\s*:?\s*([A-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ]+){0,3})',
            r'NOMBRES?\s*:?\s*([A-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ]+){0,3})',
        ]:
            m = re.search(pat, texto)
            if m:
                val = m.group(1).strip()
                if not any(p in _STOPWORDS_INE for p in val.split()):
                    datos['nombre'] = campo(val, _conf_bloque(resultado_ocr, val))
                    break

    if all(k in datos for k in ('apellido_paterno', 'apellido_materno', 'nombre')):
        nc = (f"{datos['apellido_paterno']['valor']} "
              f"{datos['apellido_materno']['valor']} "
              f"{datos['nombre']['valor']}")
        datos['nombre_completo'] = campo(nc, 0.85)

    if 'fecha_nacimiento' not in datos:
        fechas = re.findall(r'\d{2}/\d{2}/\d{4}', texto)
        if fechas and normalizar_fecha(fechas[0]):
            datos['fecha_nacimiento'] = campo(fechas[0], _conf_bloque(resultado_ocr, fechas[0]))

    curp_val, curp_conf = buscar_con_confianza(resultado_ocr, CURP_REGEX)
    if curp_val:
        curp_val = curp_val.strip()
        if len(curp_val) == 18 and validar_curp(curp_val):
            datos['curp'] = campo(curp_val, curp_conf, True)

    clave_val, clave_conf = buscar_con_confianza(resultado_ocr, r'[A-Z]{6}\d{8}[A-Z]\d{3}')
    if clave_val:
        datos['clave_elector'] = campo(clave_val, clave_conf)

    m = re.search(r'VIGENCIA\s*:?\s*(\d{4})', texto)
    if m:
        fecha_venc = f"{m.group(1)}-12-31"
        dias = calcular_dias_vencimiento(fecha_venc)
        datos['vigencia']          = campo(m.group(1), 0.9)
        datos['fecha_vencimiento'] = campo(fecha_venc, 0.9)
        datos['dias_para_vencer']  = dias
        datos['vencido']           = dias is not None and dias < 0

    m = re.search(r'SEXO\s*:?\s*(HOMBRE|MUJER|MASCULINO|FEMENINO|[HMF])\b', texto)
    if m:
        val = m.group(1).upper()
        sexo_norm = 'H' if val in ('HOMBRE', 'MASCULINO', 'H') else 'M'
        if curp_val and len(curp_val) >= 11:
            sexo_norm = 'H' if curp_val[10] == 'H' else 'M'
        datos['sexo'] = campo(sexo_norm, 0.85)
    elif curp_val and len(curp_val) >= 11:
        datos['sexo'] = campo('H' if curp_val[10] == 'H' else 'M', 0.8)

    for pat in [
        r'DOMICILIO\s*:?\s*([A-ZÁÉÍÓÚÑ0-9\s\.,#\-]+?)(?:\s{2,}|MUNICIPIO|ESTADO|$)',
        r'CALLE\s*:?\s*([A-ZÁÉÍÓÚÑ0-9\s\.,#\-]+?)(?:\s{2,}|NUM|COL|$)',
    ]:
        m = re.search(pat, texto)
        if m and len(m.group(1).strip()) > 5:
            datos['domicilio'] = campo(m.group(1).strip(), 0.75)
            break

    for label, key, pat in [
        ('MUNICIPIO', 'municipio',
         r'MUNICIPIO\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+?)(?:\s{2,}|ESTADO|ENTIDAD|$)'),
        ('ESTADO', 'estado',
         r'ESTADO\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+?)(?:\s{2,}|MUNICIPIO|$)'),
        ('LOCALIDAD', 'localidad',
         r'LOCALIDAD\s*:?\s*([A-ZÁÉÍÓÚÑ\s]+?)(?:\s{2,}|MUNICIPIO|ESTADO|$)'),
    ]:
        m = re.search(pat, texto)
        if m:
            val = m.group(1).strip()
            if len(val) > 2:
                datos[key] = campo(val, 0.75)

    return datos


# ===========================================================================
# EXTRACCIÓN — INE reverso
# ===========================================================================

def extraer_ine_reverso(resultado_ocr: OcrResult) -> dict:
    datos = {}
    texto = texto_plano(resultado_ocr)

    for pat in [r'\bOCR\s*[:\s]*(\d{9,13})\b', r'\bN(?:UM(?:ERO)?)?\s*OCR\s*[:\s]*(\d{9,13})\b']:
        m = re.search(pat, texto)
        if m:
            datos['numero_ocr'] = campo(m.group(1), 0.9)
            break

    m = re.search(r'SECCI[OÓ]N\s*[:\s]*(\d{4})', texto)
    if m:
        datos['seccion_electoral'] = campo(m.group(1), 0.85)

    m = re.search(r'A[NÑ]O\s+DE\s+REGISTRO\s*[:\s]*(\d{4})', texto)
    if m:
        datos['ano_registro'] = campo(m.group(1), 0.85)

    m = re.search(r'FOLIO\s*[:\s]*([A-Z0-9]{6,15})', texto)
    if m:
        datos['folio'] = campo(m.group(1), 0.85)

    m = re.search(r'DISTRITO\s*[:\s]*(\d{1,3})', texto)
    if m:
        datos['distrito_electoral'] = campo(m.group(1), 0.8)

    m = re.search(r'MUNICIPIO\s*[:\s]+([A-ZÁÉÍÓÚÑ\s]+?)(?:\s{2,}|ESTADO|ENTIDAD|$)', texto)
    if m and len(m.group(1).strip()) > 2:
        datos['municipio'] = campo(m.group(1).strip(), 0.75)

    m = re.search(r'ESTADO\s*[:\s]+([A-ZÁÉÍÓÚÑ\s]+?)(?:\s{2,}|MUNICIPIO|SECCION|$)', texto)
    if m and len(m.group(1).strip()) > 2:
        datos['estado'] = campo(m.group(1).strip(), 0.75)

    return datos


# ===========================================================================
# EXTRACCIÓN — generales
# ===========================================================================

def extraer_generales(resultado_ocr: OcrResult) -> dict:
    datos = {}
    curp_val, curp_conf = buscar_con_confianza(resultado_ocr, r'[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]{2}')
    if curp_val:
        datos['curp'] = campo(curp_val, curp_conf, validar_curp(curp_val))
    rfc_val, rfc_conf = buscar_con_confianza(resultado_ocr, r'[A-Z]{3,4}\d{6}[A-Z0-9]{3}')
    if rfc_val:
        datos['rfc'] = campo(rfc_val, rfc_conf, validar_rfc(rfc_val))
    return datos

# ===========================================================================
# EXTRACCIÓN — Declaración SAT
# ===========================================================================
def extraer_declaracion_sat(resultado_ocr: OcrResult) -> dict:
    """
    Extrae datos de declaraciones presentadas ante el SAT:
    - Declaración Anual (personas físicas y morales)
    - Declaraciones mensuales / provisionales IVA / ISR
    - Complementarias
 
    Campos extraídos:
        rfc, nombre_completo, tipo_declaracion, periodo_declaracion,
        ejercicio_fiscal, fecha_presentacion, numero_operacion,
        linea_captura, fecha_vigencia_linea, monto_total,
        monto_isr, monto_iva, monto_ieps, saldo_favor,
        banco_pago, resultado_declaracion
    """
    datos: dict = {}
    texto = texto_plano(resultado_ocr)
    lineas = _lineas_ordenadas(resultado_ocr)
 
    # ── RFC ──────────────────────────────────────────────────────────────────
    m = re.search(r'\bRFC[:\s]+([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})\b', texto)
    if m:
        val = m.group(1).upper()
        datos['rfc'] = campo(val, _conf_bloque(resultado_ocr, val), validar_rfc(val))
 
    # ── Nombre completo del contribuyente ────────────────────────────────────
    for pat in [
        r'CONTRIBUYENTE[:\s]+([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,60}?)(?:\s{2,}|RFC|EJERCICIO|$)',
        r'NOMBRE[:\s]+([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{5,60}?)(?:\s{2,}|RFC|EJERCICIO|$)',
        r'DENOMINACION\s*(?:O\s*RAZON\s*SOCIAL)?[:\s]+([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{3,60}?)(?:\s{2,}|RFC|$)',
    ]:
        m = re.search(pat, texto, re.IGNORECASE)
        if m:
            val = m.group(1).strip()
            if len(val) > 4:
                datos['nombre_completo'] = campo(val, _conf_bloque(resultado_ocr, val))
                break
 
    # ── Tipo de declaración ───────────────────────────────────────────────────
    tipos_declaracion = [
        'NORMAL', 'COMPLEMENTARIA', 'EXTEMPORANEA',
        'DECLARACION ANUAL', 'DECLARACION MENSUAL',
        'PROVISIONAL', 'DEFINITIVA',
    ]
    for tipo in tipos_declaracion:
        if tipo in texto:
            datos['tipo_declaracion'] = campo(tipo, 0.9)
            break
 
    # Si no se encontró por keyword, buscar por patrón
    if 'tipo_declaracion' not in datos:
        m = re.search(
            r'TIPO\s+(?:DE\s+)?DECLARACI[OÓ]N[:\s]+([A-ZÁÉÍÓÚÑ\s]+?)(?:\s{2,}|$)',
            texto, re.IGNORECASE
        )
        if m:
            datos['tipo_declaracion'] = campo(m.group(1).strip(), 0.85)
 
    # ── Ejercicio fiscal / año ────────────────────────────────────────────────
    m = re.search(r'EJERCICIO\s*(?:FISCAL)?\s*[:\s]*(20\d{2})', texto, re.IGNORECASE)
    if m:
        datos['ejercicio_fiscal'] = campo(m.group(1), 0.9)
    else:
        # Fallback: año de 4 dígitos aislado en zona de ejercicio
        m = re.search(r'\b(20\d{2})\b', texto)
        if m:
            datos['ejercicio_fiscal'] = campo(m.group(1), 0.7)
 
    # ── Periodo de declaración (mes/bimestre/trimestre) ───────────────────────
    _MESES = {
        'ENERO': '01', 'FEBRERO': '02', 'MARZO': '03', 'ABRIL': '04',
        'MAYO': '05', 'JUNIO': '06', 'JULIO': '07', 'AGOSTO': '08',
        'SEPTIEMBRE': '09', 'OCTUBRE': '10', 'NOVIEMBRE': '11', 'DICIEMBRE': '12',
    }
    m = re.search(
        r'PERIODO[:\s]+([A-ZÁÉÍÓÚÑ]+(?:\s+\d{4})?)',
        texto, re.IGNORECASE
    )
    if m:
        datos['periodo_declaracion'] = campo(m.group(1).strip(), 0.85)
    else:
        # Detectar mes en texto
        for mes_nombre, mes_num in _MESES.items():
            if mes_nombre in texto:
                datos['periodo_declaracion'] = campo(mes_nombre, 0.75)
                break
 
    # ── Número de operación / folio ───────────────────────────────────────────
    for pat in [
        r'N[UÚ]MERO\s+DE\s+OPERACI[OÓ]N[:\s]+(\d{10,20})',
        r'FOLIO\s+DE\s+PRESENTACI[OÓ]N[:\s]+([A-Z0-9]{8,20})',
        r'FOLIO[:\s]+([A-Z0-9]{8,20})',
        r'N[OÚ]\.\s*CONFIRMACI[OÓ]N[:\s]+(\d{8,15})',
    ]:
        m = re.search(pat, texto, re.IGNORECASE)
        if m:
            datos['numero_operacion'] = campo(m.group(1).strip(), 0.92)
            break
 
    # ── Fecha de presentación ─────────────────────────────────────────────────
    for pat in [
        r'FECHA\s+DE\s+PRESENTACI[OÓ]N[:\s]+(\d{2}/\d{2}/\d{4})',
        r'PRESENTADA\s+EL[:\s]+(\d{2}/\d{2}/\d{4})',
        r'FECHA\s+(?:Y\s+HORA\s+DE\s+)?ENVIO[:\s]+(\d{2}/\d{2}/\d{4})',
    ]:
        m = re.search(pat, texto, re.IGNORECASE)
        if m:
            iso = normalizar_fecha(m.group(1))
            if iso:
                datos['fecha_presentacion'] = campo(iso, 0.9)
                break
 
    # ── Línea de captura ──────────────────────────────────────────────────────
    # Formato típico: 20 dígitos o alfanumérico largo
    for pat in [
        r'L[IÍ]NEA\s+DE\s+CAPTURA[:\s]+([A-Z0-9]{15,25})',
        r'LINEA\s+CAPTURA[:\s]+([A-Z0-9]{15,25})',
        r'(?:REFERENCIA|REF)[:\s]+([A-Z0-9]{15,25})',
    ]:
        m = re.search(pat, texto, re.IGNORECASE)
        if m:
            datos['linea_captura'] = campo(m.group(1).strip(), 0.95)
            break
 
    # ── Fecha de vigencia de la línea de captura ──────────────────────────────
    for pat in [
        r'VIGENTE?\s+(?:HASTA|AL)[:\s]+(\d{2}/\d{2}/\d{4})',
        r'VIGENCIA\s+(?:DE\s+LA\s+L[IÍ]NEA)?[:\s]+(\d{2}/\d{2}/\d{4})',
        r'FECHA\s+L[IÍ]MITE\s+DE\s+PAGO[:\s]+(\d{2}/\d{2}/\d{4})',
    ]:
        m = re.search(pat, texto, re.IGNORECASE)
        if m:
            iso = normalizar_fecha(m.group(1))
            if iso:
                datos['fecha_vigencia_linea'] = campo(iso, 0.92)
                # Calcular si ya venció
                dias = calcular_dias_vencimiento(iso)
                datos['dias_para_vencer'] = dias
                datos['vencido'] = dias is not None and dias < 0
                break
 
    # ── Montos — helper interno ───────────────────────────────────────────────
    def _extraer_monto(patron: str, nombre_campo: str) -> None:
        """Busca un monto monetario y lo guarda si es válido."""
        m = re.search(patron, texto, re.IGNORECASE)
        if m:
            raw = m.group(1).replace(',', '').replace('$', '').strip()
            try:
                float(raw)   # validar que sea número
                datos[nombre_campo] = campo(raw, 0.88)
            except ValueError:
                pass
 
    # ── Monto total a pagar ───────────────────────────────────────────────────
    _extraer_monto(
        r'TOTAL\s+A\s+PAGAR[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)',
        'monto_total'
    )
    if 'monto_total' not in datos:
        _extraer_monto(
            r'IMPORTE\s+A\s+PAGAR[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)',
            'monto_total'
        )
 
    # ── ISR ───────────────────────────────────────────────────────────────────
    _extraer_monto(
        r'ISR\s+(?:A\s+CARGO|A\s+PAGAR)[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)',
        'monto_isr'
    )
    if 'monto_isr' not in datos:
        _extraer_monto(
            r'IMPUESTO\s+SOBRE\s+LA\s+RENTA[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)',
            'monto_isr'
        )
 
    # ── IVA ───────────────────────────────────────────────────────────────────
    _extraer_monto(
        r'IVA\s+(?:A\s+CARGO|A\s+PAGAR|CAUSADO)[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)',
        'monto_iva'
    )
 
    # ── IEPS ──────────────────────────────────────────────────────────────────
    _extraer_monto(
        r'IEPS[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)',
        'monto_ieps'
    )
 
    # ── Saldo a favor ─────────────────────────────────────────────────────────
    _extraer_monto(
        r'SALDO\s+A\s+FAVOR[:\s]+\$?\s*([\d,]+(?:\.\d{2})?)',
        'saldo_favor'
    )
 
    # ── Resultado de la declaración ───────────────────────────────────────────
    for resultado_kw in ['A PAGAR', 'A FAVOR', 'SALDO CERO', 'SIN IMPUESTO']:
        if resultado_kw in texto:
            datos['resultado_declaracion'] = campo(resultado_kw, 0.85)
            break
 
    # ── Banco de pago ─────────────────────────────────────────────────────────
    bancos = [
        'BBVA', 'BANAMEX', 'SANTANDER', 'BANORTE', 'HSBC', 'SCOTIABANK',
        'INBURSA', 'CITIBANAMEX', 'BANBAJIO', 'BANCOMER',
    ]
    for banco in bancos:
        if banco in texto:
            datos['banco_pago'] = campo(banco, 0.85)
            break
 
    # ── Régimen fiscal (útil para identificar tipo de contribuyente) ──────────
    m = re.search(
        r'R[EÉ]GIMEN\s+(?:FISCAL)?[:\s]+([A-ZÁÉÍÓÚÑ\s0-9]+?)(?:\s{2,}|RFC|EJERCICIO|$)',
        texto, re.IGNORECASE
    )
    if m:
        datos['regimen_fiscal'] = campo(m.group(1).strip(), 0.8)
 
    return datos

# ===========================================================================
# EXTRACCIÓN — Pasaporte
# ===========================================================================

def es_basura_pasaporte(texto: str) -> bool:
    basura = [
        'EXCLUSIVO', 'ESTADOS UNIDOS MEXICANOS', 'SECRETARIA',
        'RELACIONES EXTERIORES', 'PASAPORTE', 'MEXICO'
    ]
    return any(b in texto.upper() for b in basura)


def limpiar_nombre_pasaporte(texto: str) -> str:
    texto = texto.upper()
    basura = [
        'NACIONALIDAD', 'NATIONALITY', 'GIVEN NAMES', 'GIVEN', 'NAMES',
        'SURNAME', 'APELLIDOS', 'SEX', 'SEXO'
    ]
    for b in basura:
        texto = texto.replace(b, '')
    texto = re.sub(r'\s+', ' ', texto).strip()
    if len(texto.split()) > 4:
        texto = " ".join(texto.split()[:4])
    return texto


def extraer_pasaporte(resultado_ocr: OcrResult) -> dict:
    datos = {}
    texto = texto_plano(resultado_ocr)
    lineas = _lineas_ordenadas(resultado_ocr)

    mrz = re.findall(r'P<MEX[^\n]+', texto)
    if mrz:
        linea = mrz[0]
        nombre_partes = linea.split('<<')
        if len(nombre_partes) >= 2:
            apellidos = nombre_partes[0].replace('P<MEX', '').replace('<', ' ').strip()
            nombres   = nombre_partes[1].replace('<', ' ').strip()
            if apellidos:
                datos['apellidos'] = campo(apellidos, 0.99)
            if nombres:
                datos['nombre'] = campo(nombres, 0.99)
        m_num = re.search(r'[A-Z]\d{8}', linea)
        if m_num:
            datos['numero_pasaporte'] = campo(m_num.group(0), 0.99)

    for i, (_, txt, conf) in enumerate(lineas):
        t = txt.upper()
        if 'GIVEN NAMES' in t or 'NOMBRES' in t:
            if i + 1 < len(lineas):
                val = limpiar_nombre_pasaporte(lineas[i + 1][1])
                if len(val) > 2:
                    datos['nombre'] = campo(val, round(lineas[i + 1][2], 3))
        if 'SURNAME' in t or 'APELLIDOS' in t:
            if i + 1 < len(lineas):
                val = limpiar_nombre_pasaporte(lineas[i + 1][1])
                if len(val) > 2:
                    datos['apellidos'] = campo(val, round(lineas[i + 1][2], 3))

    for label in ['PASSPORT NO', 'NO. DE PASAPORTE', 'PASAPORTE NO', 'DOCUMENT NO']:
        m = re.search(label + r'.{0,20}?([A-Z]\d{8})', texto)
        if m:
            datos['numero_pasaporte'] = campo(m.group(1), 0.95)
            break
    if 'numero_pasaporte' not in datos:
        candidatos = re.findall(r'\b[A-Z0-9]{6,12}\b', texto)
        for c in candidatos:
            if not es_basura_pasaporte(c):
                datos['numero_pasaporte'] = campo(c, 0.7)
                break

    m = re.search(r'(NACIONALIDAD|NATIONALITY)\s*[:/]?\s*(MEXICANA?|MEX)', texto)
    if m:
        datos['nacionalidad'] = campo('MEXICANA', 0.95)

    m = re.search(r'(SEXO|SEX)\s*[:/]?\s*([MFH])', texto)
    if m:
        sexo_map = {'M': 'H', 'F': 'M', 'H': 'H'}
        datos['sexo'] = campo(sexo_map.get(m.group(2), m.group(2)), 0.9)

    for label, key in [
        (r'(FECHA DE NACIMIENTO|DATE OF BIRTH)', 'fecha_nacimiento'),
        (r'(FECHA DE EXPEDICION|DATE OF ISSUE)', 'fecha_expedicion'),
        (r'(FECHA DE CADUCIDAD|DATE OF EXPIRY|EXPIRATION)', 'fecha_vencimiento'),
    ]:
        m = re.search(label + r'.{0,60}?(\d{2}/\d{2}/\d{4})', texto)
        if m:
            fecha = normalizar_fecha(m.group(2))
            if fecha:
                datos[key] = campo(fecha, 0.9)

    if all(k not in datos for k in ['fecha_nacimiento', 'fecha_expedicion', 'fecha_vencimiento']):
        fechas_raw = re.findall(r'\d{2}[\/\-\s]\d{2}[\/\-\s]\d{4}', texto)
        fechas_norm = sorted({
            normalizar_fecha(re.sub(r'\s+', '/', f))
            for f in fechas_raw
            if normalizar_fecha(re.sub(r'\s+', '/', f))
        })
        if len(fechas_norm) >= 3:
            datos['fecha_nacimiento']  = campo(fechas_norm[0], 0.75)
            datos['fecha_expedicion']  = campo(fechas_norm[1], 0.75)
            datos['fecha_vencimiento'] = campo(fechas_norm[2], 0.75)
        elif len(fechas_norm) == 2:
            datos['fecha_expedicion']  = campo(fechas_norm[0], 0.85)
            datos['fecha_vencimiento'] = campo(fechas_norm[1], 0.85)

    curp_val, curp_conf = buscar_con_confianza(
        resultado_ocr, r'\b[A-Z][AEIOU][A-Z]{2}\d{6}[HM][A-Z]{5}[A-Z0-9]{2}\b')
    if curp_val:
        datos['curp'] = campo(curp_val, curp_conf, validar_curp(curp_val))

    if 'fecha_vencimiento' in datos:
        dias = calcular_dias_vencimiento(datos['fecha_vencimiento']['valor'])
        datos['dias_para_vencer'] = dias
        datos['vencido']          = dias is not None and dias < 0

    return datos


# ===========================================================================
# EXTRACCIÓN — Constancia SAT
# ===========================================================================

def extraer_constancia_sat(resultado_ocr: OcrResult) -> dict:
    datos = {}
    texto = texto_plano(resultado_ocr)
    lineas = _lineas_ordenadas(resultado_ocr)

    def _es_nombre_sat(txt: str) -> bool:
        palabras = txt.strip().split()
        return (
            2 <= len(palabras) <= 6
            and all(re.match(r'^[A-ZÁÉÍÓÚÑ]+$', p) for p in palabras)
            and not any(p in {'RFC', 'CURP', 'SAT', 'CIF', 'EMISION', 'ESTADO'} for p in palabras)
        )

    m = re.search(r'\bRFC[:\s]+([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})\b', texto)
    if m:
        val = m.group(1).upper()
        datos['rfc'] = campo(val, _conf_bloque(resultado_ocr, val), validar_rfc(val))

    m = re.search(r'\bID\s*CIF[:\s]*([0-9]{5,20})\b', texto, re.IGNORECASE)
    if not m:
        m = re.search(r'\bIDCIF[:\s]*([0-9]{5,20})\b', texto, re.IGNORECASE)
    if m:
        val = m.group(1)
        datos['idcif'] = campo(val, _conf_bloque(resultado_ocr, val), True)

    for i, (_, txt, conf) in enumerate(lineas):
        t = txt.upper()
        if 'NOMBRE, DENOMINACIÓN O RAZÓN SOCIAL' in t or 'NOMBRE, DENOMINACION O RAZON SOCIAL' in t:
            if i - 1 >= 0:
                _, prev_txt, prev_conf = lineas[i - 1]
                prev_txt = prev_txt.strip().upper()
                if _es_nombre_sat(prev_txt):
                    datos['nombre_completo'] = campo(prev_txt, round(prev_conf, 3))
                    break
            m_inline = re.search(
                r'([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,5})\s+NOMBRE,\s*DENOMINACI[ÓO]N O RAZ[ÓO]N SOCIAL',
                t
            )
            if m_inline:
                datos['nombre_completo'] = campo(m_inline.group(1).strip(), round(conf, 3))
                break

    if 'nombre_completo' not in datos:
        m = re.search(
            r'([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,5})\s+NOMBRE,\s*DENOMINACI[ÓO]N O RAZ[ÓO]N SOCIAL',
            texto
        )
        if m:
            datos['nombre_completo'] = campo(m.group(1).strip(), _conf_bloque(resultado_ocr, m.group(1)))

    for pat in [
        r'FECHA\s+DE\s+[ÚU]LTIMO\s+CAMBIO\s+DE\s+ESTADO\s*[:\s]+(\d{1,2}\s+DE\s+[A-ZÁÉÍÓÚÑ]+\s+DE\s+\d{4})',
        r'FECHA\s+DE\s+ULTIMO\s+CAMBIO\s+DE\s+ESTADO\s*[:\s]+(\d{1,2}\s+DE\s+[A-ZÁÉÍÓÚÑ]+\s+DE\s+\d{4})',
    ]:
        m = re.search(pat, texto, re.IGNORECASE)
        if m:
            fecha_raw = m.group(1).upper().strip()
            fecha_iso = normalizar_fecha_es_mx(fecha_raw)
            datos['fecha_ultimo_cambio_estado'] = campo(fecha_iso if fecha_iso else fecha_raw, 0.9)
            break

    m = re.search(
        r'LUGAR\s+Y\s+FECHA\s+DE\s+EMISI[ÓO]N.*?A\s+(\d{1,2}\s+DE\s+[A-ZÁÉÍÓÚÑ]+\s+DE\s+\d{4})',
        texto, re.IGNORECASE
    )
    if m:
        fecha_raw = m.group(1).upper().strip()
        fecha_iso = normalizar_fecha_es_mx(fecha_raw)
        datos['fecha_emision'] = campo(fecha_iso if fecha_iso else fecha_raw, 0.9)

    return datos


# ===========================================================================
# EXTRACCIÓN — Acta de Nacimiento
# ===========================================================================

_MESES_ES = {
    'ENERO': '01', 'FEBRERO': '02', 'MARZO': '03', 'ABRIL': '04',
    'MAYO': '05', 'JUNIO': '06', 'JULIO': '07', 'AGOSTO': '08',
    'SEPTIEMBRE': '09', 'OCTUBRE': '10', 'NOVIEMBRE': '11', 'DICIEMBRE': '12',
}

_ESTADOS_MX = {
    'AGUASCALIENTES', 'BAJA CALIFORNIA', 'BAJA CALIFORNIA SUR', 'CAMPECHE',
    'CHIAPAS', 'CHIHUAHUA', 'CIUDAD DE MEXICO', 'COAHUILA', 'COLIMA',
    'DURANGO', 'GUANAJUATO', 'GUERRERO', 'HIDALGO', 'JALISCO',
    'MEXICO', 'MICHOACAN', 'MORELOS', 'NAYARIT', 'NUEVO LEON', 'OAXACA',
    'PUEBLA', 'QUERETARO', 'QUINTANA ROO', 'SAN LUIS POTOSI', 'SINALOA',
    'SONORA', 'TABASCO', 'TAMAULIPAS', 'TLAXCALA', 'VERACRUZ',
    'YUCATAN', 'ZACATECAS',
}

_STOPWORDS_ACTA = {
    'MEXICO', 'ESTADOS', 'UNIDOS', 'REGISTRO', 'CIVIL', 'OFICIALIA',
    'LIBRO', 'ACTA', 'NACIMIENTO', 'FOJA', 'TOMO', 'CURP', 'SEXO',
    'FECHA', 'PADRE', 'MADRE', 'ABUELO', 'ABUELA', 'DIRECTOR',
    'GOBIERNO', 'SECRETARIA', 'GENERAL', 'MUNICIPIO', 'ESTADO',
    'NACIONAL', 'POBLACION', 'NOMBRE', 'NUMERO', 'NOM', 'DEL',
    'LOS', 'LAS', 'POR', 'CON', 'QUE', 'SAN', 'CLAVE', 'UNICA',
    'RENAPO', 'OFICIAL', 'FEDERATIVA', 'ENTIDAD', 'COMERCIAL',
    # Palabras SAT que no deben confundirse como nombre en acta
    'CONSTANCIA', 'SITUACION', 'FISCAL', 'CONTRIBUYENTE', 'SAT',
    'SERVICIO', 'ADMINISTRACION', 'TRIBUTARIA',
}


def _normalizar_fecha_acta(texto: str) -> str | None:
    texto = texto.upper().strip()
    for fmt in ('%d/%m/%Y', '%d-%m-%Y'):
        try:
            return datetime.strptime(texto, fmt).strftime('%Y-%m-%d')
        except ValueError:
            pass
    m = re.search(r'(\d{1,2})\s+(?:DE\s+)?([A-ZÁÉÍÓÚ]+)\s+(?:DE\s+)?(\d{4})', texto)
    if m:
        dia, mes_txt, anio = m.group(1), m.group(2), m.group(3)
        mes = _MESES_ES.get(mes_txt)
        if mes:
            return f"{anio}-{mes}-{dia.zfill(2)}"
    return None


def extraer_acta_nacimiento(resultado_ocr: OcrResult) -> dict:
    datos = {}
    texto = texto_plano(resultado_ocr)
    lineas = _lineas_ordenadas(resultado_ocr)

    m = re.search(r'FOLIO[:\s]*([A-Z0-9\-]{6,25})', texto)
    if m:
        datos['folio'] = campo(m.group(1), 0.9)

    m = re.search(r'ACTA\s+(?:NUMERO|NÚM|NO)\s*[:\s]*(\d+)', texto)
    if m:
        datos['numero_acta'] = campo(m.group(1), 0.9)

    for l in lineas:
        if 'REGISTRO' in l[1] or 'LEVANTO' in l[1]:
            fecha = _normalizar_fecha_acta(l[1])
            if fecha:
                datos['fecha_registro'] = campo(fecha, l[2])
                break

    for l in lineas:
        if 'NAC' in l[1]:
            fecha = _normalizar_fecha_acta(l[1])
            if fecha:
                datos['fecha_nacimiento'] = campo(fecha, l[2])
                break

    m = re.search(r'(MASCULINO|FEMENINO)', texto)
    if m:
        datos['sexo'] = campo('H' if 'MASCULINO' in m.group(1) else 'M', 0.9)

    for _, txt, conf in lineas:
        if 'ENTIDAD' in txt or 'ESTADO' in txt:
            m = re.search(r'(?:ENTIDAD|ESTADO)\s*[:\s]*([A-ZÁÉÍÓÚÑ\s]+)', txt)
            if m:
                val = m.group(1).strip()
                match_estado = next((e for e in _ESTADOS_MX if e in val), None)
                datos['entidad_registro'] = campo(match_estado or val, conf)
                break

    for _, txt, conf in lineas:
        if 'MUNICIPIO' in txt:
            m = re.search(r'MUNICIPIO\s*[:\s]*([A-ZÁÉÍÓÚÑ\s]+)', txt)
            if m:
                datos['municipio_registro'] = campo(m.group(1).strip(), conf)
                break

    m = re.search(r'(?:NACIO EN|LUGAR DE NACIMIENTO)\s*[:\s]*(.+)', texto)
    if m:
        datos['lugar_nacimiento'] = campo(m.group(1).strip(), 0.85)

    candidatos_nombre = []
    for (_, txt, conf) in lineas:
        palabras = txt.split()
        if (
            2 <= len(palabras) <= 6
            and all(p.isalpha() for p in palabras)
            and not any(p in _STOPWORDS_ACTA for p in palabras)
            and len(txt) >= 8
        ):
            candidatos_nombre.append((txt, conf))

    if candidatos_nombre:
        nombre_ganador = max(candidatos_nombre, key=lambda x: x[1])[0]
        datos['nombre_completo'] = campo(nombre_ganador, 0.9)

    for i, (_, txt, conf) in enumerate(lineas):
        if 'PADRE' in txt and i + 1 < len(lineas):
            sig = lineas[i + 1][1]
            if len(sig.split()) >= 2 and not any(p in _STOPWORDS_ACTA for p in sig.split()):
                datos['padre'] = campo(sig, lineas[i + 1][2])
                break

    for i, (_, txt, conf) in enumerate(lineas):
        if 'MADRE' in txt and i + 1 < len(lineas):
            sig = lineas[i + 1][1]
            if len(sig.split()) >= 2 and not any(p in _STOPWORDS_ACTA for p in sig.split()):
                datos['madre'] = campo(sig, lineas[i + 1][2])
                break

    return datos


# ===========================================================================
# EXTRACCIÓN — Formato CURP
# ===========================================================================
 
def extraer_formato_curp(resultado_ocr: OcrResult) -> dict:
    datos = {}
    texto = texto_plano(resultado_ocr)
 
    cv, cc = buscar_con_confianza(resultado_ocr, r'[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]{2}')
    if cv and validar_curp(cv):
        datos['curp'] = campo(cv, cc, True)
 
    m = re.search(r'NOMBRE\(?S?\)?\s*[:/]?\s*([A-ZÁÉÍÓÚÑ\s]{3,40}?)(?:\s{2,}|PRIMER|$)', texto)
    if m:
        datos['nombre'] = campo(m.group(1).strip(), 0.75)
 
    m = re.search(r'PRIMER\s+APELLIDO\s*[:/]?\s*([A-ZÁÉÍÓÚÑ\s]{2,30}?)(?:\s{2,}|SEGUNDO|$)', texto)
    if m:
        datos['apellido_paterno'] = campo(m.group(1).strip(), 0.75)
 
    m = re.search(r'SEGUNDO\s+APELLIDO\s*[:/]?\s*([A-ZÁÉÍÓÚÑ\s]{2,30}?)(?:\s{2,}|SEXO|FECHA|$)', texto)
    if m:
        datos['apellido_materno'] = campo(m.group(1).strip(), 0.75)
 
    for pat in [
        r'FECHA\s+DE\s+NACIMIENTO\s*[:/]?\s*(\d{2}/\d{2}/\d{4})',
        r'FECHA\s+DE\s+NACIMIENTO\s*[:/]?\s*(\d{1,2}\s+DE\s+[A-ZÁÉÍÓÚ]+\s+DE\s+\d{4})',
    ]:
        m = re.search(pat, texto)
        if m:
            datos['fecha_nacimiento'] = campo(m.group(1).strip(), 0.8)
            break
 
    m = re.search(
        r'(?:ENTIDAD|ESTADO)\s+(?:DE\s+)?(?:NACIMIENTO|REGISTRO)\s*[:/]?\s*([A-ZÁÉÍÓÚÑ\s]{4,30}?)(?:\s{2,}|$)',
        texto)
    if m:
        datos['entidad'] = campo(m.group(1).strip(), 0.75)
 
    # ── NUEVO: Fecha de certificación / emisión / impresión ────────────────
    # Se usa para detectar CURPs con más de 2 meses de antigüedad
    for pat in [
        r'FECHA\s+DE\s+CERTIFICACI[OÓ]N\s*[:/]?\s*(\d{2}/\d{2}/\d{4})',
        r'FECHA\s+DE\s+IMPRESI[OÓ]N\s*[:/]?\s*(\d{2}/\d{2}/\d{4})',
        r'FECHA\s+DE\s+EMISI[OÓ]N\s*[:/]?\s*(\d{2}/\d{2}/\d{4})',
        r'GENERADO\s+EL\s*[:/]?\s*(\d{2}/\d{2}/\d{4})',
        r'FOLIO\s+[A-Z0-9]+.*?(\d{2}/\d{2}/\d{4})',      # folio seguido de fecha
    ]:
        m = re.search(pat, texto)
        if m:
            fecha_iso = normalizar_fecha(m.group(1))
            if fecha_iso:
                datos['fecha_emision'] = campo(fecha_iso, 0.85)
                break
 
    return datos

# ===========================================================================
# CLASIFICACIÓN  ← Códigos primero (señal definitiva)
# ===========================================================================

def _detectar_tipo_por_codigos(codigos: list[dict]) -> tuple[str | None, dict]:
    """
    Inspecciona QR y CODE128 para determinar el tipo de documento
    ANTES de hacer scoring textual.
    No intercepta QR del SAT (lo deja para clasificar_por_qr_sat).
    """
    for cod in codigos:
        tipo_cod  = cod.get('tipo', '')
        datos_str = cod.get('datos', '')

        if tipo_cod == 'QRCODE':
            # QR URL del INE
            if _es_qr_url_ine(datos_str):
                return 'INE', {}

            # QR del SAT → NO interceptar aquí, dejarlo a clasificar_por_qr_sat
            # para que extraiga RFC e IDCIF correctamente
            if 'siat.sat.gob.mx' in datos_str:
                return None, {}  # fuerza el paso a clasificar_por_qr_sat

            # QR pipe-separated: ¿primer campo es CURP?
            if '|' in datos_str:
                tipo_doc, datos_qr = clasificar_qr_curp_o_acta(datos_str)
                return tipo_doc, datos_qr

        if tipo_cod == 'CODE128':
            val = datos_str.strip()
            # CODE128 que ES una CURP completa → documento CURP
            if _es_curp_valida(val):
                return 'FORMATO_CURP', {'curp': val}

    return None, {}


def clasificar_por_qr_sat(codigos: list[dict]) -> tuple[str | None, dict]:
    """
    Detecta QR del SAT (siat.sat.gob.mx) y extrae RFC e ID CIF.
    Cubre tanto la URL con 'validadorqr' como cualquier variante del dominio SAT.
    """
    datos_qr = {}
    for cod in codigos:
        datos = cod.get('datos', '')
        datos_l = datos.lower()
        # Detectar por dominio SAT o por combinación sat+validador
        es_sat_url = ('siat.sat.gob.mx' in datos_l or
                      'sat.gob.mx' in datos_l or
                      ('sat' in datos_l and 'validador' in datos_l))
        if not es_sat_url:
            continue

        # Extraer IDCIF_RFC del parámetro D3=IDCIF_RFC
        # Nota: NO usar [\d_]* antes del grupo — provoca backtracking incorrecto
        m = re.search(
            r'D3=([0-9]{5,20})_([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})',
            datos, re.IGNORECASE
        )
        if not m:
            # fallback: buscar el patrón IDCIF_RFC en cualquier parte de la URL
            m = re.search(r'(?<![0-9])([0-9]{5,20})_([A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3})\b',
                          datos, re.IGNORECASE)
        if m:
            datos_qr['idcif'] = campo(m.group(1), 0.99, True, fuente='QR')
            rfc = m.group(2).upper()
            datos_qr['rfc']   = campo(rfc, 0.99, validar_rfc(rfc), fuente='QR')
            return "CONSTANCIA_SAT", datos_qr

        # URL SAT detectada pero sin D3 parseable → aun así clasificar como SAT
        return "CONSTANCIA_SAT", {}

    return None, {}


def clasificar_documento(texto: str, codigos: list[dict] = None) -> tuple[str, dict]:
    texto = texto.upper()
    # Versión sin acentos para scoring robusto (el OCR a veces devuelve
    # "SITUACIÓN" en vez de "SITUACION", lo que rompía la detección SAT)
    texto_norm = _quitar_acentos(texto)

    # =========================================================
    # PASO 0 — Códigos primero: señal más confiable
    # =========================================================
    if codigos:
        tipo_por_codigo, datos_codigo = _detectar_tipo_por_codigos(codigos)
        if tipo_por_codigo:
            print(f"✅ Clasificación por código: {tipo_por_codigo}")
            return tipo_por_codigo, datos_codigo

        tipo_qr, datos_qr = clasificar_por_qr_sat(codigos)
        if tipo_qr:
            print(f"✅ Clasificación por QR SAT: {tipo_qr}")
            return tipo_qr, datos_qr

    # =========================================================
    # PASO 1 — Scoring textual (usa texto_norm para acentos)
    # =========================================================
    score = {
        "CONSTANCIA_SAT": 0, "FORMATO_CURP": 0, "INE": 0,
        "ACTA_NACIMIENTO": 0, "PASAPORTE": 0, "DECLARACION_SAT": 0, "OTROS": 0,
    }

    # ── SAT ──────────────────────────────────────────────────────────────────
    if "CONSTANCIA DE SITUACION FISCAL" in texto_norm:
        score["CONSTANCIA_SAT"] += 10
        if "REGIMEN FISCAL"                             in texto_norm: score["CONSTANCIA_SAT"] += 3
        if "DATOS DE IDENTIFICACION DEL CONTRIBUYENTE"  in texto_norm: score["CONSTANCIA_SAT"] += 3
        if re.search(r'\bRFC[:\s]+[A-ZN&]{3,4}\d{6}[A-Z0-9]{3}\b', texto_norm):
            score["CONSTANCIA_SAT"] += 2
        # El SAT no puede ser otro tipo
        score["ACTA_NACIMIENTO"] = -999
        score["INE"]             = -999
        score["FORMATO_CURP"]    = -999
    else:
        score["CONSTANCIA_SAT"] = -999

    # ── Acta de Nacimiento ────────────────────────────────────────────────────
    if score["CONSTANCIA_SAT"] == -999:   # solo si no es SAT
        if any(p in texto_norm for p in ["ACTA DE NACIMIENTO", "REGISTRO CIVIL", "OFICIALIA", "LIBRO", "FOJA"]):
            score["ACTA_NACIMIENTO"] += 6
            score["FORMATO_CURP"]   -= 2
        if "ACTA DE NACIMIENTO" in texto_norm:
            score["ACTA_NACIMIENTO"] += 10
        if "REGISTRO CIVIL"  in texto_norm: score["ACTA_NACIMIENTO"] += 4
        if "OFICIALIA"       in texto_norm: score["ACTA_NACIMIENTO"] += 3
        if "LIBRO" in texto_norm and "ACTA" in texto_norm: score["ACTA_NACIMIENTO"] += 3
        if re.search(r'CURP[:\s]+[A-Z]{4}\d{6}', texto_norm): score["ACTA_NACIMIENTO"] += 2
        if "NOMBRE DEL PADRE"   in texto_norm: score["ACTA_NACIMIENTO"] += 2
        if "NOMBRE DE LA MADRE" in texto_norm: score["ACTA_NACIMIENTO"] += 2

    # ── CURP formato ──────────────────────────────────────────────────────────
    if "CLAVE UNICA DE REGISTRO DE POBLACION" in texto_norm:
        score["FORMATO_CURP"] += 10
    if "SECRETARIA DE GOBERNACION" in texto_norm and "CURP" in texto_norm:
        score["FORMATO_CURP"] += 5
    if texto_norm.count("CURP") > 2 and "CONSTANCIA" not in texto_norm:
        score["FORMATO_CURP"] += 2

    # ── INE ───────────────────────────────────────────────────────────────────
    if "INSTITUTO NACIONAL ELECTORAL" in texto_norm:
        if "CREDENCIAL PARA VOTAR" in texto_norm or "CLAVE DE ELECTOR" in texto_norm:
            score["INE"] += 6
        else:
            score["INE"] -= 3

    # ── Pasaporte ─────────────────────────────────────────────────────────────
    if "PASAPORTE" in texto_norm:
        score["PASAPORTE"] += 6

    if any(p in texto_norm for p in [
        "DECLARACION ANUAL", "DECLARACION MENSUAL", "DECLARACION PROVISIONAL",
        "LINEA DE CAPTURA", "ACUSE DE RECIBO", "ACUSE DE PRESENTACION",
    ]):
        score["DECLARACION_SAT"] += 8
        score["CONSTANCIA_SAT"] -= 5   
 
    if "NUMERO DE OPERACION" in texto_norm:
        score["DECLARACION_SAT"] += 3
 
    if "FECHA LIMITE DE PAGO" in texto_norm or "VIGENTE HASTA" in texto_norm:
        score["DECLARACION_SAT"] += 3

    if "IMPORTE A PAGAR" in texto_norm or "TOTAL A PAGAR" in texto_norm:
        score["DECLARACION_SAT"] += 2

    if "ACUSE" in texto_norm and "SAT" in texto_norm:
        score["DECLARACION_SAT"] += 2

    PRIORIDAD = ["CONSTANCIA_SAT", "DECLARACION_SAT", "ACTA_NACIMIENTO", "INE", "PASAPORTE", "FORMATO_CURP", "OTROS"]
    max_score  = max(score.values())
    candidatos = [k for k, v in score.items() if v == max_score]
    tipo = next(p for p in PRIORIDAD if p in candidatos)

    if tipo == "CONSTANCIA_SAT":
        if "INSTITUTO NACIONAL ELECTORAL" in texto_norm: tipo = "INE"
        elif "ACTA DE NACIMIENTO"          in texto_norm: tipo = "ACTA_NACIMIENTO"

    if tipo == "FORMATO_CURP":
        if re.search(r'[A-Z]{3,4}\d{6}[A-Z0-9]{3}', texto_norm) \
                and "CONSTANCIA DE SITUACION FISCAL" in texto_norm:
            tipo = "CONSTANCIA_SAT"
        if "ACTA DE NACIMIENTO" in texto_norm:
            tipo = "ACTA_NACIMIENTO"

    return tipo, {}


# ===========================================================================
# PIPELINE CENTRAL  ← QR-first para TODOS los tipos
# ===========================================================================

async def _extraer_datos_qr_first(
    resultado_ocr: OcrResult,
    tipo_doc: str,
    codigos: list[dict],
    datos_qr_clasificacion: dict,
) -> tuple[dict, str, dict, str]:
    """
    1. QR (máxima confianza).
    2. OCR base.
    3. Fusión: QR sobreescribe OCR campo por campo.
    """
    datos_qr_formateados: dict = {}
    modelo_qr = 'NINGUNO'

    # ── Datos ya detectados en clasificación ──────────────────────────────────
    if datos_qr_clasificacion:
        for k, v in datos_qr_clasificacion.items():
            if isinstance(v, dict) and 'valor' in v:
                datos_qr_formateados[k] = v
            else:
                fuente = 'QR_CURP' if tipo_doc == 'FORMATO_CURP' else 'QR_CLASIFICACION'
                if k == 'curp':
                    datos_qr_formateados[k] = campo(v, 0.99, validar_curp(str(v)), fuente=fuente)
                else:
                    datos_qr_formateados[k] = campo(v, 0.99, fuente=fuente)
        if datos_qr_formateados:
            modelo_qr = 'QR_CLASIFICACION'

    # ── INE: obtener QR URL o pipe ─────────────────────────────────────────────
    if tipo_doc == "INE":
        qr_raw, modelo_qr = await obtener_datos_qr_ine(codigos)
        if qr_raw:
            datos_qr_formateados.update(datos_qr_ine_a_campos(qr_raw, modelo_qr))
            print(f"✅ INE QR [{modelo_qr}]: {list(qr_raw.keys())}")

    # ── FORMATO_CURP: enriquecer desde QR si no se hizo en clasificación ──────
    if tipo_doc == "FORMATO_CURP" and modelo_qr == 'NINGUNO':
        for cod in codigos:
            if cod.get('tipo') == 'QRCODE' and '|' in cod.get('datos', ''):
                tipo_qr_doc, datos_qr_raw = clasificar_qr_curp_o_acta(cod['datos'])
                if tipo_qr_doc == 'FORMATO_CURP':
                    datos_qr_formateados.update(datos_qr_curp_a_campos(datos_qr_raw))
                    modelo_qr = 'QR_CURP'
                    print(f"✅ CURP QR: {list(datos_qr_raw.keys())}")
                    break
            if cod.get('tipo') == 'CODE128':
                val = cod['datos'].strip()
                if _es_curp_valida(val) and 'curp' not in datos_qr_formateados:
                    datos_qr_formateados['curp'] = campo(val, 0.99, True, fuente='CODE128')
                    modelo_qr = 'CODE128'

    # ── ACTA_NACIMIENTO: QR base64/CMS ────────────────────────────────────────
    if tipo_doc == "ACTA_NACIMIENTO":
        for cod in codigos:
            if cod.get('tipo') != 'QRCODE':
                continue
            qr_raw_acta = parsear_qr_acta_base64(cod.get('datos', ''))
            if not qr_raw_acta:
                continue
            print("✅ QR ACTA decodificado:", list(qr_raw_acta.keys()))

            # Metadata del acta
            for key in ('anio_registro', 'clave_entidad', 'libro',
                        'clave_registro_civil', 'fecha_registro', 'municipio_codigo'):
                if qr_raw_acta.get(key):
                    datos_qr_formateados[key] = campo(qr_raw_acta[key], 0.99, fuente='QR_ACTA')

            # Nombre del registrado
            for key in ('nombre', 'apellido_paterno', 'apellido_materno'):
                if qr_raw_acta.get(key):
                    datos_qr_formateados[key] = campo(qr_raw_acta[key], 0.99, fuente='QR_ACTA')

            # Nombre completo derivado
            if all(qr_raw_acta.get(k) for k in ('nombre', 'apellido_paterno', 'apellido_materno')):
                nc = (f"{qr_raw_acta['nombre']} "
                      f"{qr_raw_acta['apellido_paterno']} "
                      f"{qr_raw_acta['apellido_materno']}")
                datos_qr_formateados['nombre_completo'] = campo(nc, 0.99, fuente='QR_ACTA')

            if qr_raw_acta.get('fecha_nacimiento'):
                datos_qr_formateados['fecha_nacimiento'] = campo(
                    qr_raw_acta['fecha_nacimiento'], 0.99, fuente='QR_ACTA')

            if qr_raw_acta.get('sexo'):
                datos_qr_formateados['sexo'] = campo(
                    qr_raw_acta['sexo'], 0.99, fuente='QR_ACTA')

            if qr_raw_acta.get('curp'):
                cv = qr_raw_acta['curp']
                datos_qr_formateados['curp'] = campo(cv, 0.99, validar_curp(cv), fuente='QR_ACTA')

            # Padres
            if qr_raw_acta.get('padre'):
                datos_qr_formateados['padre'] = campo(qr_raw_acta['padre'], 0.99, fuente='QR_ACTA')
            if qr_raw_acta.get('madre'):
                datos_qr_formateados['madre'] = campo(qr_raw_acta['madre'], 0.99, fuente='QR_ACTA')

            modelo_qr = 'QR_ACTA'
            break

    # ── OCR base ──────────────────────────────────────────────────────────────
    datos_ocr: dict = {}

    if tipo_doc == "INE":
        datos_ocr.update(extraer_ine_frente(resultado_ocr))
        datos_ocr.update(extraer_ine_reverso(resultado_ocr))
    elif tipo_doc == "CONSTANCIA_SAT":
        datos_ocr.update(extraer_constancia_sat(resultado_ocr))
        datos_ocr.update(extraer_generales(resultado_ocr))
    elif tipo_doc == "DECLARACION_SAT":
        datos_ocr.update(extraer_declaracion_sat(resultado_ocr))
        datos_ocr.update(extraer_generales(resultado_ocr))
    elif tipo_doc == "PASAPORTE":
        datos_ocr.update(extraer_pasaporte(resultado_ocr))
    elif tipo_doc == "FORMATO_CURP":
        datos_ocr.update(extraer_formato_curp(resultado_ocr))
        datos_ocr.update(extraer_generales(resultado_ocr))
    elif tipo_doc == "ACTA_NACIMIENTO":
        datos_ocr.update(extraer_acta_nacimiento(resultado_ocr))
        # Número de acta desde CODE128 (solo si NO es una CURP)
        if 'numero_acta' not in datos_ocr:
            for cod in codigos:
                if cod.get('tipo') == 'CODE128':
                    info = parsear_code128_acta(cod['datos'])
                    if info.get('numero_acta_codigo'):
                        datos_ocr['numero_acta'] = campo(
                            info['numero_acta_codigo'], 0.9, fuente='CODE128')
                    if info.get('clave_renapo'):
                        datos_ocr.setdefault('clave_renapo', campo(
                            info['clave_renapo'], 0.9, fuente='CODE128'))
                    if info.get('curp_code128'):
                        datos_ocr.setdefault('curp', campo(
                            info['curp_code128'], 0.99,
                            validar_curp(info['curp_code128']), fuente='CODE128'))
                    break

    # ── Fusión (QR gana) ──────────────────────────────────────────────────────
    datos_finales    = combinar_datos_qr_ocr(datos_ocr, datos_qr_formateados)
    fuente_principal = "QR" if datos_qr_formateados else "OCR"

    return datos_finales, fuente_principal, datos_qr_formateados, modelo_qr

async def _guardar_historial(doc_id: str, uid: str, evento: str, detalle: str):
    """Guarda un evento en historial_documentos usando el cliente de supabase_service."""
    try:
        from supabase_service import supabase as sb
        sb.table("historial_documentos").insert({
            "uid_usuario": uid,
            "doc_id":      doc_id,
            "evento":      evento,
            "detalle":     detalle,
        }).execute()
    except Exception as e:
        print(f"⚠ historial error: {e}")
 
 
async def _actualizar_documento(doc_id: str, uid: str, campos: dict):
    """Actualiza el registro de documentos con los datos procesados."""
    try:
        from supabase_service import supabase as sb
        sb.table("documentos").update(campos).eq("id", doc_id).execute()
    except Exception as e:
        print(f"⚠ update documento error: {e}")
 

# ===========================================================================
# RESUMEN — Gemini
# ===========================================================================

async def generar_resumen_gemini(tipo_doc: str, datos: dict, texto_ocr: str) -> str:
    if not GEMINI_API_KEY:
        return "Resumen no disponible — configura GEMINI_API_KEY en tu archivo .env"

    datos_legibles = [
        f"- {k}: {v['valor']}" if isinstance(v, dict) and 'valor' in v else f"- {k}: {v}"
        for k, v in datos.items() if not isinstance(v, dict) or 'valor' in v
    ]
    prompt = f"""Eres el asistente de DocuManager, una plataforma de gestión documental mexicana.
Se analizó un documento oficial con los siguientes datos extraídos:

Tipo de documento: {tipo_doc}
Datos:
{chr(10).join(datos_legibles) if datos_legibles else '(Sin datos detectados)'}

Escribe un resumen breve (máximo 3 oraciones) en español, en lenguaje simple, para el usuario final.
Explica qué es el documento, para qué sirve, y si hay alguna alerta importante (vencimiento próximo o ya vencido).
No uses términos técnicos como OCR, API, regex o confianza.
Responde SOLO con el resumen, sin introducción ni conclusión."""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 200, "temperature": 0.3},
            })
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except httpx.HTTPStatusError as e:
        return f"Resumen no disponible temporalmente (error {e.response.status_code})"
    except Exception as e:
        print(f"⚠ Error Gemini: {e}")
        return "Resumen no disponible temporalmente"


# ===========================================================================
# n8n
# ===========================================================================
def calcular_estado_vencimiento(datos: dict, tipo_doc: str = "") -> dict:
    dias    = datos.get('dias_para_vencer')
    vencido = datos.get('vencido', False)
 
    if vencido:
        return {"estado": "VENCIDO", "alerta": True, "urgencia": "alta"}
 
    if dias is not None and dias <= DIAS_ALERTA_VENCIMIENTO:
        return {"estado": "PROXIMO_VENCER", "alerta": True, "urgencia": "media", "dias_restantes": dias}
 
    if dias is not None:
        return {"estado": "VIGENTE", "alerta": False, "dias_restantes": dias}
 
    # ── Sin fecha de vencimiento detectada ────────────────────────────────
    # Caso especial: CURP — verificar antigüedad de la emisión (≥ 60 días = vencida)
    if tipo_doc == "FORMATO_CURP":
        fecha_em_raw = datos.get("fecha_emision")
        if fecha_em_raw:
            fe_val = (fecha_em_raw.get("valor")
                      if isinstance(fecha_em_raw, dict) else fecha_em_raw)
            if fe_val:
                try:
                    fe_date = datetime.strptime(str(fe_val), '%Y-%m-%d')
                    dias_desde_emision = (datetime.now() - fe_date).days
                    if dias_desde_emision >= 60:
                        return {
                            "estado": "VENCIDO",
                            "alerta": True,
                            "urgencia": "alta",
                            "info_adicional": (
                                "Se recomienda usar una CURP descargada recientemente desde el portal "
                                "oficial para asegurar que esté certificada y actualizada. Algunas "
                                "instituciones pueden rechazar versiones antiguas."
                            ),
                        }
                except Exception:
                    pass
 
        # CURP sin fecha detectada: siempre recomendar actualizar
        return {
            "estado": "VIGENTE",
            "alerta": False,
            "info_adicional": (
                "Se recomienda usar una CURP descargada recientemente desde el portal oficial "
                "para asegurar que esté certificada y actualizada. Algunas instituciones pueden "
                "rechazar versiones antiguas."
            ),
        }
 
    # Caso general: sin fecha → VIGENTE con recomendación de mantener actualizado
    return {
        "estado": "VIGENTE",
        "alerta": False,
        "info_adicional": (
            "Este documento no tiene fecha de vencimiento registrada. "
            "Se recomienda mantenerlo actualizado y con fecha de expedición "
            "reciente para su mejor aceptación en trámites."
        ),
    }


async def notificar_n8n(payload: dict) -> None:
    if not N8N_WEBHOOK_URL:
        return
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(N8N_WEBHOOK_URL, json=payload)
            print(f"✅ n8n [{resp.status_code}] — {payload.get('documento', {}).get('tipo')}")
    except Exception as e:
        print(f"⚠ Error n8n: {e}")


def construir_payload_n8n(tipo_doc, datos, calidad, filename, resumen, 
                           uid="", email_usuario="", nombre_usuario="", 
                           apellido_usuario="") -> dict:
    vencimiento = calcular_estado_vencimiento(datos, tipo_doc)
    datos_planos = {
        k: v['valor'] if isinstance(v, dict) and 'valor' in v else v
        for k, v in datos.items()
        if not isinstance(v, dict) or 'valor' in v
    }
    return {
        "evento":    "documento_analizado",
        "timestamp": datetime.now().isoformat(),
        "usuario": {
            "uid":      uid,
            "email":    email_usuario,
            "nombre":   nombre_usuario,
            "apellido": apellido_usuario,
            "nombre_completo": f"{nombre_usuario} {apellido_usuario}".strip(),
        },
        "documento": {
            "tipo":        tipo_doc,
            "filename":    filename,
            "datos":       datos_planos,
            "resumen_ia":  resumen,
            "vencimiento": vencimiento,
        },
        "calidad_imagen": {
            "suficiente":   calidad.get("calidad_suficiente"),
            "advertencias": calidad.get("advertencias", []),
        },
        "requiere_revision_humana": not datos_planos.get("curp"),
        "alerta_vencimiento":       vencimiento.get("alerta", False),
        "procesado_exitoso":        True,
    }

def construir_respuesta(tipo_doc, datos, calidad, filename, codigos, resumen="",
                        fuente_principal="OCR") -> dict:
    curp_ok     = datos.get('curp', {}).get('valido', False) if isinstance(datos.get('curp'), dict) else False
    vencimiento = calcular_estado_vencimiento(datos, tipo_doc) 
    return {
        "documento_detectado": tipo_doc,
        "resumen":             resumen,
        "data":                datos,
        "vencimiento":         vencimiento,
        "codigos_detectados":  codigos,
        "metadata": {
            "filename":                 filename,
            "calidad_imagen":           calidad,
            "fuente_datos_principal":   fuente_principal,
            "requiere_revision_humana": not curp_ok,
        },
    }

# ===========================================================================
# HELPER Supabase token
# ===========================================================================

async def verificar_token(authorization: str) -> dict:
    if not SUPABASE_DISPONIBLE:
        raise HTTPException(503, "Supabase no configurado")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Token requerido")
    jwt_token = authorization.split(" ", 1)[1]
    try:
        return verificar_token_supabase(jwt_token)
    except Exception:
        raise HTTPException(401, "Token inválido o expirado")


# ===========================================================================
# ENDPOINTS
# ===========================================================================

@app.post("/api/ocr/analizar-documento")
async def analizar_documento(
    archivo: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    try:
        contenido   = await archivo.read()
        imagen, cal = preprocesar(contenido, archivo.filename)
        ocr_result  = reader.readtext(imagen, detail=1)
        texto       = texto_plano(ocr_result)

        codigos                  = leer_codigos(contenido, archivo.filename)
        tipo_doc, datos_qr_clas  = clasificar_documento(texto, codigos)

        datos, fuente, datos_qr, modelo_qr = await _extraer_datos_qr_first(
            ocr_result, tipo_doc, codigos, datos_qr_clas)

        resumen = await generar_resumen_groq(tipo_doc, datos, texto)

        if N8N_WEBHOOK_URL:
            background_tasks.add_task(
                notificar_n8n,
                construir_payload_n8n(tipo_doc, datos, cal, archivo.filename, resumen),
            )
        return construir_respuesta(tipo_doc, datos, cal, archivo.filename, codigos, resumen, fuente)

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        print(f"❌ {e}")
        raise HTTPException(500, str(e))


from ine_ocr_v2 import extraer_ine_frente_v2, extraer_ine_reverso_v2, combinar_ine_v2


@app.post("/api/ocr/analizar-ine")
async def analizar_ine(
    frente: UploadFile = File(...),
    reverso: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    try:
        b_frente  = await frente.read()
        b_reverso = await reverso.read()

        # ── Preprocesado específico para INE ──────────────────────────────
        img_f, cal_f = preprocesar_ine(b_frente,  frente.filename)
        img_r, cal_r = preprocesar_ine(b_reverso, reverso.filename)

        # ── OCR + limpieza (limpiar_ocr es clave para INE) ────────────────
        ocr_f_raw = reader.readtext(img_f, detail=1)
        ocr_r_raw = reader.readtext(img_r, detail=1)
        ocr_f     = limpiar_ocr(ocr_f_raw)
        ocr_r     = limpiar_ocr(ocr_r_raw)

        print("\n===== TEXTO FRENTE =====")
        print(texto_plano(ocr_f))
        print("\n===== TEXTO REVERSO =====")
        print(texto_plano(ocr_r))

        # ── Códigos QR/barras ─────────────────────────────────────────────
        codigos_f = leer_codigos(b_frente,  frente.filename)
        codigos_r = leer_codigos(b_reverso, reverso.filename)
        todos     = codigos_f + codigos_r

        # ── QR INE ────────────────────────────────────────────────────────
        qr_raw, modelo_qr = await obtener_datos_qr_ine(todos)
        datos_qr = datos_qr_ine_a_campos(qr_raw, modelo_qr) if qr_raw else {}

        if not datos_qr and qr_raw:
            datos_qr = {
                "qr_info": campo(str(qr_raw), 0.95, fuente=f"QR_{modelo_qr}")
            }

        # ── Extracción OCR ────────────────────────────────────────────────
        datos_frente_ocr  = extraer_ine_frente_v2(ocr_f)
        datos_reverso_ocr = extraer_ine_reverso_v2(ocr_r)

        # Fallback: si v2 no extrajo nombre, intentar con extractor base
        if 'nombre_completo' not in datos_frente_ocr and 'apellido_paterno' not in datos_frente_ocr:
            print("⚠ v2 sin nombre — intentando extractor base")
            datos_frente_ocr.update(extraer_ine_frente(ocr_f))

        # ── Combinación final ─────────────────────────────────────────────
        datos = combinar_ine_v2(datos_frente_ocr, datos_reverso_ocr, datos_qr)

        if not datos:
            datos = {
                "error":      campo("No se pudo extraer información", 0.0),
                "sugerencia": campo("Mejorar calidad de imagen o tomar foto más cercana", 0.0),
            }

        fuente_ppal = f"QR_{modelo_qr}" if datos_qr else "OCR"

        # ── Resumen ───────────────────────────────────────────────────────
        texto_comb = texto_plano(ocr_f) + " " + texto_plano(ocr_r)
        resumen    = await generar_resumen_groq("INE", datos, texto_comb)

        # ── Calidad combinada ─────────────────────────────────────────────
        cal_comb = {
            "calidad_suficiente": cal_f["calidad_suficiente"] and cal_r["calidad_suficiente"],
            "advertencias":       cal_f["advertencias"] + cal_r["advertencias"],
        }

        # ── n8n ───────────────────────────────────────────────────────────
        if N8N_WEBHOOK_URL:
            background_tasks.add_task(
                notificar_n8n,
                construir_payload_n8n("INE", datos, cal_comb, frente.filename, resumen),
            )

        curp_ok     = isinstance(datos.get('curp'), dict) and datos['curp'].get('valido', False)
        vencimiento = calcular_estado_vencimiento(datos, "INE")

        return {
            "documento_detectado": "INE",
            "resumen":             resumen,
            "data":                datos,
            "vencimiento":         vencimiento,
            "datos_qr_raw":        qr_raw,
            "codigos_detectados":  todos,
            "metadata": {
                "filename_frente":          frente.filename,
                "filename_reverso":         reverso.filename,
                "calidad_frente":           cal_f,
                "calidad_reverso":          cal_r,
                "modelo_qr":                modelo_qr,
                "fuente_datos_principal":   fuente_ppal,
                "requiere_revision_humana": not curp_ok,
            },
        }

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        print(f"❌ ERROR analizar-ine: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/ocr/analizar-desde-storage")
async def analizar_desde_storage(
    background_tasks: BackgroundTasks,
    url_storage: str = "",
    filename: str = "",
    authorization: Optional[str] = Header(None),
):
    usuario = await verificar_token(authorization)
    uid     = usuario["uid"]

    if not url_storage:
        raise HTTPException(422, "url_storage requerido")
    if not filename:
        filename = url_storage.split("/")[-1]

    try:
        contenido   = await descargar_archivo_storage(url_storage)
        imagen, cal = preprocesar(contenido, filename)
        ocr_result  = reader.readtext(imagen, detail=1)
        texto       = texto_plano(ocr_result)

        codigos                  = leer_codigos(contenido, filename)
        tipo_doc, datos_qr_clas  = clasificar_documento(texto, codigos)

        datos, fuente, _, _ = await _extraer_datos_qr_first(
            ocr_result, tipo_doc, codigos, datos_qr_clas)

        resumen     = await generar_resumen_gemini(tipo_doc, datos, texto)
        vencimiento = calcular_estado_vencimiento(datos, tipo_doc)
        curp_ok     = datos.get("curp", {}).get("valido", False) if isinstance(datos.get("curp"), dict) else False

        doc_id = await guardar_resultado_documento(
            uid_usuario=uid, filename=filename,
            storage_path=url_storage, url_archivo=url_storage,
            tipo_doc=tipo_doc, datos=datos, resumen=resumen,
            vencimiento=vencimiento, calidad=cal,
            requiere_revision=not curp_ok,
        )

        if N8N_WEBHOOK_URL and vencimiento.get("alerta"):
            pn = construir_payload_n8n(tipo_doc, datos, cal, filename, resumen)
            pn["doc_id"] = doc_id
            background_tasks.add_task(notificar_n8n, pn)

        return {
            "doc_id": doc_id,
            "documento_detectado": tipo_doc,
            "resumen": resumen,
            "data": datos,
            "vencimiento": vencimiento,
            "metadata": {
                "filename": filename,
                "calidad_imagen": cal,
                "fuente_datos_principal": fuente,
                "requiere_revision_humana": not curp_ok,
            },
        }

    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        print(f"❌ {e}")
        raise HTTPException(500, str(e))


@app.get("/api/ocr/mis-documentos")
async def mis_documentos(authorization: Optional[str] = Header(None)):
    usuario = await verificar_token(authorization)
    docs    = await obtener_documentos_usuario(usuario["uid"])
    return {"documentos": docs, "total": len(docs)}


@app.get("/api/ocr/documento/{doc_id}")
async def detalle_documento(doc_id: str, authorization: Optional[str] = Header(None)):
    usuario = await verificar_token(authorization)
    doc     = await obtener_documento(doc_id, usuario["uid"])
    if not doc:
        raise HTTPException(404, "Documento no encontrado")
    return doc


@app.get("/api/v2/admin/estadisticas")
async def estadisticas(authorization: Optional[str] = Header(None)):
    usuario = await verificar_token(authorization)
    rol     = await obtener_rol_usuario(usuario["uid"])
    if rol != "admin":
        raise HTTPException(403, "Acceso restringido a administradores")
    return await obtener_estadisticas()

# ── Schemas ───────────────────────────────────────────────────────────────────
 
class MensajeHistorial(BaseModel):
    rol: str          # "user" o "assistant"
    contenido: str
    ts: Optional[str] = None
 
 
class ChatRequest(BaseModel):
    mensaje:              str
    historial:            List[MensajeHistorial] = []
    contexto_documentos:  str = ""          # bloque de texto con los documentos del usuario
    nombres_docs:         List[str] = []    # nombres de archivo para mencionar en fuentes
 
 
# ── Endpoint /api/chat ────────────────────────────────────────────────────────
 
@app.post("/api/chat")
async def chat(
    body:          ChatRequest,
    authorization: Optional[str] = Header(None),
):
    """
    Chatbot que responde preguntas del usuario usando:
    - Contexto de sus documentos (datos_extraidos + resumen_ia)
    - Historial de la conversación
    - Groq (Llama 3) como LLM
    - Sugerencias proactivas por tipo de documento
    """
    if SUPABASE_DISPONIBLE:
        await verificar_token(authorization)
 
    # ── Bloque de sugerencias por tipo de documento ───────────────────────
    sugerencias_bloque = (
        "\n\nTienes acceso a las siguientes sugerencias de seguridad y buenas prácticas "
        "por tipo de documento. Incorpóralas de forma natural en tu respuesta cuando el "
        "usuario pregunte sobre ese tipo de documento o cuando detectes que es relevante. "
        "No las repitas todas; elige la más pertinente al contexto.\n"
    )
    for tipo, sugs in SUGERENCIAS_DOC.items():
        if tipo == "GENERAL":
            continue
        sugerencias_bloque += f"\n{tipo}:\n" + "\n".join(f"  - {s}" for s in sugs)
 
    sugerencias_bloque += "\n\nConsideraciones generales (aplican a todos los documentos):\n"
    sugerencias_bloque += "\n".join(f"  - {s}" for s in SUGERENCIAS_DOC["GENERAL"])
 
    # ── System prompt ─────────────────────────────────────────────────────
    system_prompt = (
        "Eres DocuBot, el asistente inteligente de DocuManager, "
        "una plataforma de gestión de documentos personales mexicanos.\n\n"
        "Tu función es ayudar al usuario a consultar información de sus documentos "
        "(INE, pasaporte, CURP, acta de nacimiento, constancia fiscal, etc.).\n\n"
        "Reglas:\n"
        "- Responde SIEMPRE en español, de forma clara, breve y amable.\n"
        "- Si el usuario pregunta por un dato que está en el contexto, dalo con precisión.\n"
        "- Si el dato NO está en el contexto, dilo con honestidad: "
        "\"No encontré ese dato en tus documentos\".\n"
        "- Nunca inventes datos, nombres, fechas o CURPs.\n"
        "- Si detectas un documento próximo a vencer o ya vencido, menciónalo proactivamente.\n"
        "- Al final de cada respuesta, añade UNA sola sugerencia relevante del bloque de "
        "sugerencias, precedida por 💡 *Sugerencia:*. Si no hay sugerencia aplicable, omítela.\n"
        "- Usa **negrita** solo para resaltar datos importantes (nombres, fechas, CURPs, etc.).\n"
        "- Máximo 4 oraciones por respuesta a menos que el usuario pida más detalle.\n"
        "- No uses listas de bullet points; responde en prosa natural."
        + sugerencias_bloque
    )
 
    # ── Contexto de documentos ────────────────────────────────────────────
    if body.contexto_documentos.strip():
        contexto_bloque = (
            f"\nA continuación están los documentos del usuario con sus datos extraídos:\n\n"
            f"{body.contexto_documentos}\n\n"
            "---\n"
            "Usa esta información para responder la pregunta del usuario.\n"
            "Si la pregunta no tiene relación con los documentos, igual puedes ayudar de forma general."
        )
    else:
        contexto_bloque = "El usuario no tiene documentos cargados en este momento."
 
    # ── Historial ─────────────────────────────────────────────────────────
    messages = [
        {"role": "system", "content": system_prompt + "\n\n" + contexto_bloque}
    ]
    for msg in body.historial[-8:]:
        role = "user" if msg.rol == "user" else "assistant"
        messages.append({"role": role, "content": msg.contenido})
    messages.append({"role": "user", "content": body.mensaje})
 
    # ── Llamar a Groq ─────────────────────────────────────────────────────
    try:
        from groq_service import groq_client
 
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        respuesta_texto = response.choices[0].message.content.strip()
 
    except Exception as e:
        print(f"❌ Error Groq chat: {e}")
        raise HTTPException(500, "Error al generar respuesta. Intenta de nuevo.")
 
    # ── Fuentes mencionadas ───────────────────────────────────────────────
    fuentes_mencionadas = []
    for nombre in body.nombres_docs:
        nombre_base = nombre.replace("_", " ").lower()
        if any(
            part in respuesta_texto.lower() or part in body.contexto_documentos.lower()
            for part in nombre_base.split()[:2]
        ):
            fuentes_mencionadas.append(nombre)
    fuentes_mencionadas = fuentes_mencionadas[:3]
 
    return {
        "respuesta": respuesta_texto,
        "fuentes":   fuentes_mencionadas,
    }

@app.post("/api/ocr/analizar-y-actualizar")
async def analizar_y_actualizar(
    background_tasks: BackgroundTasks,
    archivo:          UploadFile = File(...),
    x_doc_id:         Optional[str] = Header(None),
    x_uid:            Optional[str] = Header(None),
    x_email:          Optional[str] = Header(None),
    x_nombre:         Optional[str] = Header(None),
    x_apellido:       Optional[str] = Header(None),
    authorization:    Optional[str] = Header(None),
):
    """
    Procesa OCR de un documento y actualiza el registro existente en Supabase.
    Se llama en background desde Next.js — no bloquea al usuario.
    
    NUEVO: verifica si el documento fue cancelado antes de procesar y antes de
    actualizar, para no pisarle encima a una cancelación del usuario.
    """
    doc_id = x_doc_id
    uid    = x_uid
 
    # ── Verificación temprana: ¿ya fue cancelado/eliminado? ───────────────
    try:
        from supabase_service import supabase as sb
        check = sb.table("documentos") \
            .select("eliminado_en, estado") \
            .eq("id", doc_id) \
            .single() \
            .execute()
 
        if check.data:
            fue_eliminado  = check.data.get("eliminado_en") is not None
            fue_cancelado  = check.data.get("estado") == "cancelado"
            if fue_eliminado or fue_cancelado:
                print(f"⚠ Doc {doc_id} cancelado por el usuario — abortando OCR")
                return {"ok": False, "reason": "cancelled"}
    except Exception as e:
        # Si no podemos consultar, seguimos (mejor procesar que bloquear)
        print(f"⚠ No se pudo verificar estado de cancelación: {e}")
 
    try:
        contenido = await archivo.read()
        filename  = archivo.filename or "documento"
 
        # ── Preprocesamiento ──────────────────────────────────────────────
        imagen, cal = preprocesar(contenido, filename)
 
        # ── OCR en thread separado ─────────────────────────────────────────
        async with ocr_semaphore:
            ocr_result = await asyncio.to_thread(reader.readtext, imagen, detail=1)
        texto = texto_plano(ocr_result)
 
        # ── Segunda verificación: ¿fue cancelado MIENTRAS corría el OCR? ──
        try:
            check2 = sb.table("documentos") \
                .select("eliminado_en, estado") \
                .eq("id", doc_id) \
                .single() \
                .execute()
            if check2.data:
                if check2.data.get("eliminado_en") or check2.data.get("estado") == "cancelado":
                    print(f"⚠ Doc {doc_id} cancelado durante OCR — descartando resultado")
                    return {"ok": False, "reason": "cancelled_during_ocr"}
        except Exception:
            pass
 
        # ── Clasificar y extraer ───────────────────────────────────────────
        codigos                 = leer_codigos(contenido, filename)
        tipo_doc, datos_qr_clas = clasificar_documento(texto, codigos)
 
        datos, fuente, datos_qr, modelo_qr = await _extraer_datos_qr_first(
            ocr_result, tipo_doc, codigos, datos_qr_clas
        )
 
        # ── Resumen Groq ───────────────────────────────────────────────────
        resumen     = await generar_resumen_groq(tipo_doc, datos, texto)
        vencimiento = calcular_estado_vencimiento(datos, tipo_doc)
        curp_ok     = isinstance(datos.get("curp"), dict) and datos["curp"].get("valido", False)
 
        # ── Extraer fecha vencimiento ──────────────────────────────────────
        fecha_venc = None
        if datos.get("fecha_vencimiento"):
            fv = datos["fecha_vencimiento"]
            fecha_venc = fv["valor"] if isinstance(fv, dict) else fv
 
        # ── Actualizar registro en DB ──────────────────────────────────────
        await _actualizar_documento(doc_id, uid, {
            "tipo_doc":           tipo_doc,
            "datos_extraidos":    datos,
            "calidad_imagen":     cal,
            "resumen_ia":         resumen,
            "vencimiento_estado": vencimiento.get("estado", "SIN_FECHA"),
            "vencimiento_alerta": vencimiento.get("alerta", False),
            "dias_para_vencer":   vencimiento.get("dias_restantes"),
            "fecha_vencimiento":  fecha_venc,
            "requiere_revision":  not curp_ok,
            "estado":             "procesado",
            "actualizado_en":     datetime.now().isoformat(),
            "info_adicional":     vencimiento.get("info_adicional", ""),
        })
 
        # ── Historial ──────────────────────────────────────────────────────
        await _guardar_historial(
            doc_id, uid, "procesado",
            f"OCR completado — {tipo_doc} — fuente: {fuente}"
        )
 
        # ── n8n background ─────────────────────────────────────────────────
        if N8N_WEBHOOK_URL:
            email_usuario    = x_email    or ""
            nombre_usuario   = x_nombre   or ""
            apellido_usuario = x_apellido or ""
 
            # Enriquecer desde DB solo si los headers estaban vacíos
            if not email_usuario:
                try:
                    res = sb.table("usuarios").select("email, nombre, apellido") \
                        .eq("id", uid).single().execute()
                    if res.data:
                        email_usuario    = res.data.get("email", "")
                        nombre_usuario   = res.data.get("nombre", "")
                        apellido_usuario = res.data.get("apellido", "")
                except Exception as e:
                    print(f"⚠ No se pudo obtener datos del usuario: {e}")
 
            background_tasks.add_task(
                notificar_n8n,
                construir_payload_n8n(
                    tipo_doc, datos, cal, filename, resumen,
                    uid=uid,
                    email_usuario=email_usuario,
                    nombre_usuario=nombre_usuario,
                    apellido_usuario=apellido_usuario,
                ),
            )
 
        return {"ok": True, "tipo_doc": tipo_doc, "doc_id": doc_id}
 
    except Exception as e:
        print(f"❌ analizar-y-actualizar: {e}")
        if doc_id:
            await _actualizar_documento(doc_id, uid, {
                "estado":    "error",
                "resumen_ia": f"Error en procesamiento: {str(e)[:200]}",
            })
            await _guardar_historial(doc_id, uid, "error", f"Error OCR: {str(e)[:200]}")
        raise HTTPException(500, str(e))
 
 
# ── Endpoint: analizar INE (frente+reverso) y actualizar ────────────────────
@app.post("/api/ocr/analizar-ine-y-actualizar")
async def analizar_ine_y_actualizar(
    background_tasks: BackgroundTasks,
    frente:        UploadFile = File(...),
    reverso:       UploadFile = File(...),
    x_doc_id:      Optional[str] = Header(None),
    x_uid:         Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Procesa OCR de INE (frente + reverso) en paralelo y actualiza el registro.
    """
    doc_id = x_doc_id
    uid    = x_uid
 
    try:
        b_frente  = await frente.read()
        b_reverso = await reverso.read()
 
        img_f, cal_f = preprocesar_ine(b_frente,  frente.filename or "frente.jpg")
        img_r, cal_r = preprocesar_ine(b_reverso, reverso.filename or "reverso.jpg")
 
        # ── OCR en paralelo (¡la clave de la velocidad!) ───────────────────
        ocr_f, ocr_r = await asyncio.gather(
            asyncio.to_thread(reader.readtext, img_f, detail=1),
            asyncio.to_thread(reader.readtext, img_r, detail=1),
        )
 
        codigos_f = leer_codigos(b_frente,  frente.filename or "frente.jpg")
        codigos_r = leer_codigos(b_reverso, reverso.filename or "reverso.jpg")
        todos     = codigos_f + codigos_r
 
        qr_raw, modelo_qr = await obtener_datos_qr_ine(todos)
        datos_qr          = datos_qr_ine_a_campos(qr_raw, modelo_qr) if qr_raw else {}
 
        from ine_ocr_v2 import extraer_ine_frente_v2, extraer_ine_reverso_v2, combinar_ine_v2
 
        # Extracción OCR también en paralelo
        datos_f, datos_r = await asyncio.gather(
            asyncio.to_thread(extraer_ine_frente_v2,  ocr_f),
            asyncio.to_thread(extraer_ine_reverso_v2, ocr_r),
        )
 
        datos       = combinar_ine_v2(datos_f, datos_r, datos_qr)
        fuente_ppal = f"QR_{modelo_qr}" if qr_raw else "OCR"
 
        texto_comb  = texto_plano(ocr_f) + " " + texto_plano(ocr_r)
        resumen     = await generar_resumen_groq("INE", datos, texto_comb)
        vencimiento = calcular_estado_vencimiento(datos, "INE")
        curp_ok     = isinstance(datos.get("curp"), dict) and datos["curp"].get("valido", False)
 
        fecha_venc = None
        if datos.get("fecha_vencimiento"):
            fv = datos["fecha_vencimiento"]
            fecha_venc = fv["valor"] if isinstance(fv, dict) else fv
 
        cal_comb = {
            "calidad_suficiente": cal_f["calidad_suficiente"] and cal_r["calidad_suficiente"],
            "advertencias":       cal_f["advertencias"] + cal_r["advertencias"],
        }
 
        await _actualizar_documento(doc_id, uid, {
            "tipo_doc":          "INE",
            "datos_extraidos":   datos,
            "calidad_imagen":    cal_comb,
            "resumen_ia":        resumen,
            "vencimiento_estado": vencimiento.get("estado", "SIN_FECHA"),
            "vencimiento_alerta": vencimiento.get("alerta", False),
            "dias_para_vencer":  vencimiento.get("dias_restantes"),
            "fecha_vencimiento": fecha_venc,
            "requiere_revision": not curp_ok,
            "estado":            "procesado",
            "actualizado_en":    datetime.now().isoformat(),
            "info_adicional": vencimiento.get("info_adicional", ""),
        })
 
        await _guardar_historial(
            doc_id, uid, "procesado",
            f"INE procesada — fuente: {fuente_ppal} — QR modelo: {modelo_qr}"
        )
 
        if N8N_WEBHOOK_URL:
            email_usuario    = ""
            nombre_usuario   = ""
            apellido_usuario = ""
            try:
                from supabase_service import supabase as sb
                res = sb.table("usuarios").select(
                    "email, nombre, apellido"
                ).eq("id", uid).single().execute()
                if res.data:
                    email_usuario    = res.data.get("email", "")
                    nombre_usuario   = res.data.get("nombre", "")
                    apellido_usuario = res.data.get("apellido", "")
            except Exception as e:
                print(f"⚠ No se pudo obtener datos del usuario: {e}")

            background_tasks.add_task(
                notificar_n8n,
                construir_payload_n8n(
                    "INE", datos, cal_comb, frente.filename, resumen,
                    uid=uid,
                    email_usuario=email_usuario,
                    nombre_usuario=nombre_usuario,
                    apellido_usuario=apellido_usuario,
                ),
            )
 
        return {"ok": True, "tipo_doc": "INE", "doc_id": doc_id}
 
    except Exception as e:
        print(f"❌ analizar-ine-y-actualizar: {e}")
        if doc_id:
            await _actualizar_documento(doc_id, uid, {"estado": "error"})
            await _guardar_historial(doc_id, uid, "error", f"Error OCR INE: {str(e)[:200]}")
        raise HTTPException(500, str(e))

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "3.5",
        "gemini_configurado": bool(GEMINI_API_KEY),
        "n8n_configurado": bool(N8N_WEBHOOK_URL),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)