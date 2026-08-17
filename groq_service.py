import os
from groq import Groq

_GROQ_KEY  = os.getenv("GROQ_API_KEY", "")
groq_client = Groq(api_key=_GROQ_KEY) if _GROQ_KEY else None

# 1. Separamos el comportamiento del modelo (SYSTEM)
INSTRUCCIONES_SISTEMA_DERRAME = """\
Eres el asistente de DocuManager, sistema de gestión documental del Aeropuerto Internacional de Querétaro (AIQ).
Escribe un resumen en español de máximo 7 oraciones, claro y directo, para el personal administrativo. El resumen debe:
1. Identificar la aerolínea responsable, la aeronave (matrícula), la fecha/hora y la ubicación del incidente.
2. Describir el tipo de sustancia derramada (combustible) y el volumen.
3. Mencionar medidas de contención y si la fecha límite de pago está próxima o vencida.
No uses términos técnicos. Responde SOLO con el resumen; sin conclusión.\
"""

# 2. Separamos los datos a inyectar (USER)
def construir_mensaje_usuario(datos: dict) -> str:
    lineas = []
    for k, v in datos.items():
        if isinstance(v, dict) and 'valor' in v:
            lineas.append(f"- {k}: {v['valor']}")
        elif not isinstance(v, dict):
            lineas.append(f"- {k}: {v}")
    datos_fmt = '\n'.join(lineas) if lineas else '(Sin datos detectados)'
    return f"Analiza el siguiente Reporte de Derrame y genera el resumen:\n\n{datos_fmt}"

async def generar_resumen_groq(tipo_doc: str, datos: dict, texto_ocr: str) -> str:
    if not groq_client:
        return "DEBUG_ERROR: Falta configurar GROQ_API_KEY en .env"

    if tipo_doc == "REPORTE_DERRAME":
        msg_system = INSTRUCCIONES_SISTEMA_DERRAME
        msg_user = construir_mensaje_usuario(datos)
    else:
        msg_system = "Eres el asistente de SmartDocs. Resume este documento en máximo 4 oraciones, explicando qué es y si hay alertas. No uses lenguaje técnico."
        datos_legibles = [
            f"- {k}: {v['valor']}" if isinstance(v, dict) and 'valor' in v else f"- {k}: {v}"
            for k, v in datos.items()
            if not isinstance(v, dict) or 'valor' in v
        ]
        msg_user = f"Tipo de documento: {tipo_doc}\nDatos:\n{chr(10).join(datos_legibles)}"

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=[
                {"role": "system", "content": msg_system},
                {"role": "user", "content": msg_user}
            ],
            # Eliminamos max_tokens temporalmente para evitar que el modelo corte la respuesta
            temperature=0.3,
        )
        
        resumen = response.choices[0].message.content.strip()
        
        if not resumen:
            return "DEBUG_ERROR: Groq respondió con éxito, pero el texto devuelto está en blanco."
            
        return resumen

    except Exception as e:
        print(f"⚠ Error Groq resumen: {e}")
        return f"DEBUG_ERROR: Fallo en la API ({str(e)})"