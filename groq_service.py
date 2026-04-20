"""
groq_service.py — cliente Groq seguro (no falla si falta la API key)
"""
import os
from groq import Groq

_GROQ_KEY  = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=_GROQ_KEY) if _GROQ_KEY else None


async def generar_resumen_groq(tipo_doc: str, datos: dict, texto_ocr: str) -> str:
    if not groq_client:
        return "Resumen no disponible — configura GROQ_API_KEY en .env"

    datos_legibles = [
        f"- {k}: {v['valor']}" if isinstance(v, dict) and 'valor' in v else f"- {k}: {v}"
        for k, v in datos.items()
        if not isinstance(v, dict) or 'valor' in v
    ]

    prompt = f"""Eres el asistente de SmartDocs, una plataforma de gestión documental mexicana.
Se analizó un documento oficial con los siguientes datos extraídos:

Tipo de documento: {tipo_doc}
Datos:
{chr(10).join(datos_legibles) if datos_legibles else '(Sin datos detectados)'}

Escribe un resumen breve (máximo 4 oraciones) en español, en lenguaje simple, para el usuario final.
Explica qué es el documento, para qué sirve, y si hay alguna alerta importante (vencimiento próximo o vencido).
No uses términos técnicos como OCR, API, regex o confianza.
Responde SOLO con el resumen, sin introducción ni conclusión."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠ Error Groq resumen: {e}")
        return "Resumen no disponible temporalmente"