from __future__ import annotations

import re
import numpy as np
from datetime import datetime
from typing import Optional
import httpx

OcrResult = list[tuple]

# =============================================================================
# PRIMITIVOS
# =============================================================================

def _c(valor, conf: float, valido: bool | None = None, fuente: str = "ocr") -> dict:
    r: dict = {"valor": valor, "confianza": round(float(conf), 3), "fuente": fuente}
    if valido is not None:
        r["valido"] = valido
    return r


def _lineas(ocr: OcrResult) -> list[tuple[float, float, str, float]]:
    out = []
    for bbox, txt, conf in ocr:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        out.append((float(np.mean(ys)), float(min(xs)), txt.upper().strip(), float(conf)))
    out.sort(key=lambda r: r[0])
    return out


def _txt(ocr: OcrResult) -> str:
    return " ".join(t.upper().strip() for _, t, _ in ocr)


def _conf_de(ocr: OcrResult, token: str) -> float:
    cs = [c for _, t, c in ocr if token.upper() in t.upper()]
    return float(np.mean(cs)) if cs else 0.70


# =============================================================================
# VALIDACIONES
# =============================================================================

_CURP_RE = re.compile(
    r'\b([A-Z][AEIOU][A-Z]{2}'
    r'\d{6}'
    r'[HM]'
    r'[A-Z]{2}'
    r'[BCDFGHJKLMNPQRSTVWXYZ]{3}'
    r'[A-Z0-9]\d)\b'
)
_CLAVE_RE = re.compile(r'\b([A-Z]{6}\d{8}[A-Z]\d{3})\b')


def _validar_curp(curp: str) -> bool:
    if len(curp) != 18:
        return False
    if not re.match(
        r'^[A-Z][AEIOU][A-Z]{2}\d{6}[HM][A-Z]{2}'
        r'[BCDFGHJKLMNPQRSTVWXYZ]{3}[A-Z0-9]\d$', curp):
        return False
    try:
        datetime.strptime(curp[4:10], '%y%m%d')
    except ValueError:
        return False
    return True


def _fecha_iso(s: str) -> Optional[str]:
    s = re.sub(r'[\s\-]', '/', s.strip())
    for fmt in ('%d/%m/%Y', '%d/%m/%y', '%Y/%m/%d'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


def _dias_vence(fecha_iso: str) -> Optional[int]:
    try:
        return (datetime.strptime(fecha_iso, '%Y-%m-%d') - datetime.now()).days
    except Exception:
        return None


# =============================================================================
# STOPWORDS
# =============================================================================

_STOP: set[str] = {
    'INSTITUTO', 'NACIONAL', 'ELECTORAL', 'CREDENCIAL', 'PARA', 'VOTAR',
    'NOMBRE', 'APELLIDO', 'CURP', 'CLAVE', 'ELECTOR', 'VIGENCIA',
    'DOMICILIO', 'MUNICIPIO', 'ESTADO', 'SECCION', 'LOCALIDAD',
    'DISTRITO', 'FECHA', 'NACIMIENTO', 'REGISTRO', 'MEXICO', 'EDAD',
    'SEXO', 'FIRMA', 'ANO', 'AÑO', 'EMISION', 'EMISIÓN', 'FOLIO',
    'INE', 'IFE', 'PATERNO', 'MATERNO', 'NOMBRES', 'DELEGACION',
    'DELEGACIÓN', 'CALLE', 'COL', 'COLONIA', 'NUM', 'NUMERO', 'NÚMERO',
    'DE', 'LA', 'EL', 'LOS', 'LAS', 'DEL',
}

_ETIQUETAS_AP_PAT = [
    'APELLIDO PATERNO', 'PRIMER APELLIDO', 'AP. PATERNO', 'AP PATERNO', 'PATERNO'
]
_ETIQUETAS_AP_MAT = [
    'APELLIDO MATERNO', 'SEGUNDO APELLIDO', 'AP. MATERNO', 'AP MATERNO', 'MATERNO'
]
_ETIQUETAS_NOMBRE = [
    'NOMBRE(S):', 'NOMBRE(S)', 'NOMBRES:', 'NOMBRES', 'NOMBRE:', 'NOMBRE'
]
_ETIQUETAS_FEC_NAC = [
    'FECHA DE NACIMIENTO', 'F. NACIMIENTO', 'F.NACIMIENTO', 'FECHA NAC', 'NACIMIENTO'
]
_ETIQUETAS_VIGENCIA = ['VIGENCIA:', 'VIGENCIA']
_ETIQUETAS_SECCION  = ['SECCION:', 'SECCIÓN:', 'SECCION', 'SECCIÓN']


def _es_nombre(txt: str) -> bool:
    palabras = [p for p in txt.strip().split() if p]
    if not (1 <= len(palabras) <= 4):
        return False
    if any(p in _STOP for p in palabras):
        return False
    if not all(re.match(r'^[A-ZÁÉÍÓÚÑÜ]+$', p) for p in palabras):
        return False
    if any(len(p) < 2 for p in palabras):
        return False
    return True


# =============================================================================
# DEBUG
# =============================================================================

def debug_ocr(ocr: OcrResult, titulo: str = "OCR DEBUG"):
    print(f"\n===== {titulo} =====")
    for i, (bbox, txt, conf) in enumerate(ocr):
        print(f"{i:02d} | {conf:.2f} | {txt}")


def limpiar_ocr(ocr_result: OcrResult) -> OcrResult:
    limpio = []
    for bbox, txt, conf in ocr_result:
        txt = txt.strip()
        if conf < 0.30:
            continue
        if len(txt) < 2:
            continue
        if not any(c.isalnum() for c in txt):
            continue
        limpio.append((bbox, txt, conf))
    return limpio


# =============================================================================
# EXTRACCIÓN — FRENTE INE  (versión robusta con múltiples estrategias)
# =============================================================================

def extraer_ine_frente_v2(ocr: OcrResult) -> dict:
    """
    Extrae datos del FRENTE de la INE con 3 estrategias en cascada:
      1. Búsqueda por etiquetas posicionales (más precisa)
      2. Búsqueda por regex en texto plano
      3. Heurística por bloques de nombre (fallback)

    Campos: apellido_paterno, apellido_materno, nombre, nombre_completo,
            fecha_nacimiento, curp, clave_elector, vigencia,
            fecha_vencimiento, dias_para_vencer, vencido, sexo
    """
    datos: dict = {}
    lineas_ord = _lineas(ocr)
    texto = _txt(ocr)

    # ── ESTRATEGIA 1: etiquetas posicionales ─────────────────────────────────
    i = 0
    while i < len(lineas_ord):
        _, _, txt, conf = lineas_ord[i]

        # Apellido paterno
        if any(e in txt for e in _ETIQUETAS_AP_PAT) and 'apellido_paterno' not in datos:
            for j in range(i + 1, min(i + 4, len(lineas_ord))):
                _, _, sig, cs = lineas_ord[j]
                # El valor puede estar en la misma línea tras la etiqueta
                resto = sig
                for e in _ETIQUETAS_AP_PAT:
                    resto = resto.replace(e, '').strip().strip(':').strip()
                if _es_nombre(resto):
                    datos['apellido_paterno'] = _c(resto, round(cs, 3))
                    break
                # O el siguiente bloque puede ser el valor
                if _es_nombre(sig) and j == i + 1:
                    datos['apellido_paterno'] = _c(sig, round(cs, 3))
                    break
            i += 1; continue

        # Apellido materno
        if any(e in txt for e in _ETIQUETAS_AP_MAT) and 'apellido_materno' not in datos:
            for j in range(i + 1, min(i + 4, len(lineas_ord))):
                _, _, sig, cs = lineas_ord[j]
                resto = sig
                for e in _ETIQUETAS_AP_MAT:
                    resto = resto.replace(e, '').strip().strip(':').strip()
                if _es_nombre(resto):
                    datos['apellido_materno'] = _c(resto, round(cs, 3))
                    break
                if _es_nombre(sig) and j == i + 1:
                    datos['apellido_materno'] = _c(sig, round(cs, 3))
                    break
            i += 1; continue

        # Nombre(s)
        if any(e in txt for e in _ETIQUETAS_NOMBRE) and 'nombre' not in datos:
            # El valor puede venir en el mismo bloque (a la derecha de NOMBRE:)
            resto = txt
            for e in _ETIQUETAS_NOMBRE:
                resto = resto.replace(e, '').strip().strip(':').strip()
            if _es_nombre(resto) and len(resto) > 1:
                datos['nombre'] = _c(resto, round(conf, 3))
            elif i + 1 < len(lineas_ord):
                _, _, sig, cs = lineas_ord[i + 1]
                if _es_nombre(sig):
                    datos['nombre'] = _c(sig, round(cs, 3))
            i += 1; continue

        # Fecha de nacimiento
        if any(e in txt for e in _ETIQUETAS_FEC_NAC) and 'fecha_nacimiento' not in datos:
            fm = re.search(r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})', txt)
            if fm:
                iso = _fecha_iso(fm.group(1))
                if iso:
                    datos['fecha_nacimiento'] = _c(iso, round(conf, 3))
            elif i + 1 < len(lineas_ord):
                _, _, sig, cs = lineas_ord[i + 1]
                fm2 = re.search(r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})', sig)
                if fm2:
                    iso = _fecha_iso(fm2.group(1))
                    if iso:
                        datos['fecha_nacimiento'] = _c(iso, round(cs, 3))
            i += 1; continue

        # Vigencia
        if any(e in txt for e in _ETIQUETAS_VIGENCIA) and 'vigencia' not in datos:
            mv = re.search(r'(\d{4})', txt)
            if mv:
                _agregar_vigencia(datos, mv.group(1), conf)
            elif i + 1 < len(lineas_ord):
                _, _, sig, cs = lineas_ord[i + 1]
                mv2 = re.search(r'(\d{4})', sig)
                if mv2:
                    _agregar_vigencia(datos, mv2.group(1), cs)
            i += 1; continue

        i += 1

    # ── ESTRATEGIA 2: regex en texto plano ───────────────────────────────────
    if 'apellido_paterno' not in datos:
        m = re.search(
            r'APELLIDO\s+PATERNO\s*:?\s*([A-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ]+){0,2})', texto)
        if m:
            datos['apellido_paterno'] = _c(m.group(1).strip(), _conf_de(ocr, m.group(1)))

    if 'apellido_materno' not in datos:
        m = re.search(
            r'APELLIDO\s+MATERNO\s*:?\s*([A-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ]+){0,2})', texto)
        if m:
            datos['apellido_materno'] = _c(m.group(1).strip(), _conf_de(ocr, m.group(1)))

    if 'nombre' not in datos:
        for pat in [
            r'NOMBRE\(?S?\)?\s*:?\s*([A-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ]+){0,3})',
            r'NOMBRES?\s*:?\s*([A-ZÁÉÍÓÚÑ]+(?:\s+[A-ZÁÉÍÓÚÑ]+){0,3})',
        ]:
            m = re.search(pat, texto)
            if m:
                val = m.group(1).strip()
                if _es_nombre(val):
                    datos['nombre'] = _c(val, _conf_de(ocr, val))
                    break

    if 'fecha_nacimiento' not in datos:
        fechas = re.findall(r'\d{2}/\d{2}/\d{4}', texto)
        for f in fechas:
            iso = _fecha_iso(f)
            if iso:
                datos['fecha_nacimiento'] = _c(iso, _conf_de(ocr, f))
                break

    if 'vigencia' not in datos:
        m = re.search(r'VIGENCIA\s*:?\s*(\d{4})', texto)
        if m:
            _agregar_vigencia(datos, m.group(1), 0.85)

    # ── ESTRATEGIA 3: heurística por bloques (fallback nombre) ───────────────
    if 'apellido_paterno' not in datos or 'nombre' not in datos:
        _heuristica_nombre(lineas_ord, datos)

    # ── CURP ─────────────────────────────────────────────────────────────────
    if 'curp' not in datos:
        m = _CURP_RE.search(texto)
        if m:
            curp = m.group(1)
            datos['curp'] = _c(curp, _conf_de(ocr, curp), _validar_curp(curp))

    # ── Clave de elector ─────────────────────────────────────────────────────
    if 'clave_elector' not in datos:
        m = _CLAVE_RE.search(texto)
        if m:
            datos['clave_elector'] = _c(m.group(1), _conf_de(ocr, m.group(1)))

    # ── Sexo (desde CURP o texto) ─────────────────────────────────────────────
    if 'sexo' not in datos:
        curp_val = datos.get('curp', {}).get('valor', '') if isinstance(datos.get('curp'), dict) else ''
        m = re.search(r'\bSEXO\s*:?\s*(HOMBRE|MUJER|MASCULINO|FEMENINO|[HMF])\b', texto)
        if m:
            sv = m.group(1).upper()
            sexo = 'H' if sv in ('HOMBRE', 'MASCULINO', 'H') else 'M'
            if curp_val and len(curp_val) >= 11:
                sexo = 'H' if curp_val[10] == 'H' else 'M'
            datos['sexo'] = _c(sexo, 0.85)
        elif curp_val and len(curp_val) >= 11:
            datos['sexo'] = _c('H' if curp_val[10] == 'H' else 'M', 0.80)

    # ── Nombre completo compuesto ─────────────────────────────────────────────
    if 'nombre_completo' not in datos:
        partes = [
            datos.get('apellido_paterno', {}).get('valor', ''),
            datos.get('apellido_materno', {}).get('valor', ''),
            datos.get('nombre', {}).get('valor', ''),
        ]
        if all(partes):
            confs = [
                datos.get('apellido_paterno', {}).get('confianza', 0.7),
                datos.get('apellido_materno', {}).get('confianza', 0.7),
                datos.get('nombre', {}).get('confianza', 0.7),
            ]
            datos['nombre_completo'] = _c(' '.join(partes), round(min(confs), 3))

    return datos


def _agregar_vigencia(datos: dict, year_str: str, conf: float) -> None:
    """Helper: agrega vigencia + fecha_vencimiento + dias_para_vencer + vencido."""
    try:
        year = int(year_str)
        if year < 2000 or year > 2060:
            return
        fecha_venc = f"{year}-12-31"
        dias = (datetime.strptime(fecha_venc, '%Y-%m-%d') - datetime.now()).days
        datos['vigencia']          = _c(year_str, round(conf, 3))
        datos['fecha_vencimiento'] = _c(fecha_venc, round(conf, 3))
        datos['dias_para_vencer']  = dias
        datos['vencido']           = dias < 0
    except Exception:
        pass


def _heuristica_nombre(
    lineas: list[tuple[float, float, str, float]],
    datos: dict,
) -> None:
    """
    Fallback: recoge bloques que parecen nombre/apellido y los asigna
    solo a los campos que aún faltan.
    """
    candidatos = [
        (txt, conf)
        for _, _, txt, conf in lineas
        if _es_nombre(txt) and len(txt) >= 3
    ]
    # Ordenar por posición (ya vienen ordenados por Y) → tomamos en orden de aparición
    faltan = [k for k in ('apellido_paterno', 'apellido_materno', 'nombre') if k not in datos]
    for k, (txt, conf) in zip(faltan, candidatos):
        datos[k] = _c(txt, round(conf, 3))


# =============================================================================
# EXTRACCIÓN — REVERSO INE
# =============================================================================

def extraer_ine_reverso_v2(ocr: OcrResult) -> dict:
    """
    Extrae campos del REVERSO de la INE.
    Campos: numero_ocr, seccion_electoral, ano_registro,
            numero_emision, folio, distrito_electoral, municipio, estado, curp
    """
    datos: dict = {}
    txt = _txt(ocr)

    # Número OCR
    for pat in [
        r'\bOCR\s*[:/]?\s*(\d{9,13})\b',
        r'\bN[UÚ]M(?:ERO)?\s+(?:DE\s+)?OCR\s*[:/]?\s*(\d{9,13})\b',
        r'\b(\d{13})\b',
    ]:
        m = re.search(pat, txt)
        if m:
            datos['numero_ocr'] = _c(m.group(1), 0.88)
            break

    # Sección electoral
    m = re.search(r'SECC\w{0,5}\s*[:/]?\s*(\d{3,4})', txt)
    if m:
        datos['seccion_electoral'] = _c(m.group(1), 0.85)

    # Año de registro
    m = re.search(r'(?:A[NÑ]O\s+DE\s+REGISTRO|A\w{0,3}\s+DE\s+REGIS\w*)\s*[:/]?\s*(\d{4})', txt)
    if m:
        datos['ano_registro'] = _c(m.group(1), 0.85)

    # Número de emisión
    m = re.search(r'(?:N[UÚ]M(?:ERO)?\s+DE\s+)?EMISI[OÓ]N\s*[:/]?\s*(\d{1,2})\b', txt)
    if m:
        datos['numero_emision'] = _c(m.group(1), 0.85)

    # Folio
    m = re.search(r'\bFOLIO\s*[:/]?\s*([A-Z0-9]{6,15})\b', txt)
    if m:
        datos['folio'] = _c(m.group(1), 0.85)

    # Distrito
    m = re.search(r'\bDISTRITO\s*[:/]?\s*(\d{1,3})\b', txt)
    if m:
        datos['distrito_electoral'] = _c(m.group(1), 0.80)

    # Municipio
    m = re.search(r'MUNICIPIO\s*[:/]?\s*([A-ZÁÉÍÓÚÑ\s]+?)(?=\s{2,}|ESTADO|ENTIDAD|\Z)', txt)
    if m and len(m.group(1).strip()) > 2:
        datos['municipio'] = _c(m.group(1).strip(), 0.72)

    # Estado
    m = re.search(r'\bESTADO\s*[:/]?\s*([A-ZÁÉÍÓÚÑ\s]+?)(?=\s{2,}|MUNICIPIO|\Z)', txt)
    if m and len(m.group(1).strip()) > 2:
        datos['estado'] = _c(m.group(1).strip(), 0.72)

    # CURP (algunos modelos de reverso la incluyen)
    m = _CURP_RE.search(txt)
    if m:
        curp = m.group(1)
        datos['curp'] = _c(curp, _conf_de(ocr, curp), _validar_curp(curp))

    return datos


# =============================================================================
# COMBINACIÓN FRENTE + REVERSO + QR
# =============================================================================

def combinar_ine_v2(
    datos_frente: dict,
    datos_reverso: dict,
    datos_qr: dict | None = None,
) -> dict:
    """
    Fusiona frente + reverso + QR.
    Prioridad: QR (0.99) > OCR frente > OCR reverso.
    """
    resultado = dict(datos_frente)

    for k, v in datos_reverso.items():
        if k not in resultado:
            resultado[k] = v

    if datos_qr:
        resultado.update(datos_qr)

    return resultado


# =============================================================================
# QR INE
# =============================================================================

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

            print(f"✅ QR INE URL — campos: {list(datos.keys())}")

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


def parsear_qr_url_ine(url: str) -> dict:
    partes = url.strip().split('/')
    try:
        return {
            "id_ine": partes[-4],
            "fecha_emision": partes[-3],
            "tipo_qr": partes[-2],
            "codigo_qr": partes[-1],
        }
    except Exception:
        return {}


async def obtener_datos_qr_ine(codigos: list[dict]) -> tuple[dict, str]:
    qr_url  = None
    qr_pipe = None

    print("\n🧾 QR detectados:")
    for cod in codigos:
        print(type(cod.get("datos")), str(cod.get("datos"))[:80])

    for cod in codigos:
        if cod.get('tipo') != 'QRCODE':
            continue
        datos = cod.get('datos')
        if not isinstance(datos, str):
            continue
        datos = datos.strip()
        if any(ord(c) < 32 for c in datos[:20]):
            continue
        if _es_qr_url_ine(datos):
            qr_url = datos
            continue
        if '|' in datos and len(datos) < 300:
            qr_pipe = datos

    if qr_url:
        raw = await _fetch_datos_qr_ine_url(qr_url)
        if not raw:
            print("⚠ Usando fallback parseo URL QR")
            raw = parsear_qr_url_ine(qr_url)
        return raw, 'URL_2023'

    if qr_pipe:
        return parsear_qr_ine_pipe(qr_pipe), 'PIPE_2022'

    return {}, 'NINGUNO'


def _validar_curp_ext(curp: str) -> bool:
    return _validar_curp(curp)


def datos_qr_ine_a_campos(raw: dict, modelo: str = 'PIPE_2022') -> dict:
    fuente = f"QR_{modelo}"
    datos  = {}

    def _qc(val, valido=None):
        r = {"valor": val, "confianza": 0.99, "fuente": fuente}
        if valido is not None:
            r["valido"] = valido
        return r

    if raw.get('curp'):
        datos['curp'] = _qc(raw['curp'], _validar_curp(raw['curp']))

    for key in ('apellido_paterno', 'apellido_materno', 'nombre',
                'nombre_completo', 'sexo', 'fecha_nacimiento',
                'clave_elector', 'numero_ocr', 'numero_emision'):
        if raw.get(key):
            datos[key] = _qc(raw[key])

    if 'nombre_completo' not in datos:
        partes = [raw.get(k, '') for k in ('apellido_paterno', 'apellido_materno', 'nombre')]
        if all(partes):
            datos['nombre_completo'] = _qc(' '.join(partes))

    if raw.get('vigencia'):
        year      = raw['vigencia']
        fecha_venc = f"{year}-12-31"
        try:
            dias = (datetime.strptime(fecha_venc, '%Y-%m-%d') - datetime.now()).days
        except Exception:
            dias = None
        datos['vigencia']          = _qc(year)
        datos['fecha_vencimiento'] = _qc(fecha_venc)
        datos['dias_para_vencer']  = dias
        datos['vencido']           = dias is not None and dias < 0

    # Para modelo URL_2023 — datos de emisión del QR
    if modelo == 'URL_2023':
        if raw.get('fecha_emision'):
            datos['fecha_emision'] = _qc(raw['fecha_emision'])
        if raw.get('id_ine'):
            datos['id_ine'] = _qc(raw['id_ine'])

    return datos