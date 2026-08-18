from supabase import create_client, Client
import os
import json
import httpx
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Faltan SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY en .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────
# GUARDAR DOCUMENTO OCR
# ─────────────────────────────────────────────
async def guardar_resultado_documento(
    uid_usuario: str,
    filename: str,
    storage_path: str,
    url_archivo: str,
    tipo_doc: str,
    datos: dict,
    resumen: str,
    vencimiento: dict,
    calidad: dict,
    requiere_revision: bool,
) -> str:
    # Extraer fecha de vencimiento si existe
    fecha_venc = None
    fecha_raw = datos.get("fecha_vencimiento", {})
    if isinstance(fecha_raw, dict):
        fecha_venc = fecha_raw.get("valor")

    payload = {
        "uid_usuario":        uid_usuario,
        "filename":           filename,
        "storage_path":       storage_path,
        "url_archivo":        url_archivo,
        "tipo_doc":           tipo_doc,
        # Serializar correctamente para JSONB
        "datos_extraidos":    json.loads(json.dumps(datos,   default=str)),
        "calidad_imagen":     json.loads(json.dumps(calidad, default=str)),
        "resumen_ia":         resumen,
        # Vencimiento
        "vencimiento_estado": vencimiento.get("estado", "VIGENTE"),   # antes era "SIN_FECHA"
        "vencimiento_alerta": vencimiento.get("alerta", False),
        "dias_para_vencer":   vencimiento.get("dias_restantes"),
        "fecha_vencimiento":  fecha_venc,
        # ── NUEVO: información adicional / recomendación ──────────────────
        "info_adicional":     vencimiento.get("info_adicional", ""),
        # Control
        "requiere_revision":  requiere_revision,
        "estado":             "procesado",
    }

    res = supabase.table("documentos").insert(payload).execute()
    return res.data[0]["id"]


# ─────────────────────────────────────────────
# OBTENER DOCUMENTOS DE UN USUARIO
# ─────────────────────────────────────────────
async def obtener_documentos_usuario(uid: str) -> list:
    res = (
        supabase.table("documentos")
        .select("*")
        .eq("uid_usuario", uid)
        .order("creado_en", desc=True)
        .execute()
    )
    return res.data


# ─────────────────────────────────────────────
# OBTENER UN DOCUMENTO
# ─────────────────────────────────────────────
async def obtener_documento(doc_id: str, uid: str) -> dict | None:
    res = (
        supabase.table("documentos")
        .select("*")
        .eq("id", doc_id)
        .eq("uid_usuario", uid)
        .single()
        .execute()
    )
    return res.data


# ─────────────────────────────────────────────
# ESTADÍSTICAS (admin)
# ─────────────────────────────────────────────
async def obtener_estadisticas() -> dict:
    res  = supabase.table("documentos").select("tipo_doc, vencimiento_estado").execute()
    docs = res.data

    por_tipo = {}
    vencidos = 0

    for doc in docs:
        tipo = doc.get("tipo_doc", "OTROS")
        por_tipo[tipo] = por_tipo.get(tipo, 0) + 1
        if doc.get("vencimiento_estado") == "VENCIDO":
            vencidos += 1

    return {
        "total_documentos": len(docs),
        "por_tipo":         por_tipo,
        "vencidos":         vencidos,
    }


# ─────────────────────────────────────────────
# MARCAR COMO REVISADO
# ─────────────────────────────────────────────
async def marcar_revisado(doc_id: str) -> None:
    supabase.table("documentos").update({
        "requiere_revision": False,
        "revisado_en":       datetime.now().isoformat(),
    }).eq("id", doc_id).execute()


# ─────────────────────────────────────────────
# ROL DE USUARIO
# ─────────────────────────────────────────────
async def obtener_rol_usuario(uid: str) -> str:
    res = (
        supabase.table("usuarios")
        .select("rol")
        .eq("id", uid)
        .single()
        .execute()
    )
    return res.data.get("rol", "usuario") if res.data else "usuario"


# ─────────────────────────────────────────────
# VERIFICAR JWT DE SUPABASE AUTH
# ─────────────────────────────────────────────
def verificar_token_supabase(jwt: str) -> dict:
    user = supabase.auth.get_user(jwt)
    return {
        "uid":   user.user.id,
        "email": user.user.email,
    }


# ─────────────────────────────────────────────
# DESCARGAR ARCHIVO DESDE STORAGE
# ─────────────────────────────────────────────
async def descargar_archivo_storage(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.get(url)
        res.raise_for_status()
        return res.content

# Agrega esto al final de supabase_service.py:

def get_supabase_client() -> Client:
    return supabase