import asyncio
from supabase_service import supabase

async def test_conexion():
    print("🔌 Probando conexión a Supabase...\n")

    # 1. Verificar conexión básica
    try:
        res = supabase.table("usuarios").select("id, nombre, email, rol").execute()
        print(f"✅ Conexión OK — {len(res.data)} usuarios encontrados:")
        for u in res.data:
            print(f"   • {u['nombre']} ({u['email']}) — rol: {u['rol']}")
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        return

    print()

    # 2. Verificar tabla documentos
    try:
        res = supabase.table("documentos").select("id, tipo_doc, estado").limit(5).execute()
        print(f"✅ Tabla documentos OK — {len(res.data)} registros")
    except Exception as e:
        print(f"❌ Error en tabla documentos: {e}")

    print()

    # 3. Verificar columnas clave
    try:
        res = supabase.rpc("version").execute()  # simple ping
        print(f"✅ Supabase responde correctamente")
    except Exception as e:
        print(f"⚠️  Ping opcional falló (no crítico): {e}")

    print()

    # 4. Insertar documento de prueba y borrarlo
    try:
        uid_prueba = "d0b0083a-0321-4d78-9bcb-94ef2783313a"  # Maria (ya existe)

        insert = supabase.table("documentos").insert({
            "uid_usuario":       uid_prueba,
            "filename":          "test_conexion.pdf",
            "tipo_doc":          "OTROS",
            "estado":            "procesado",
            "datos_extraidos":   {},
            "calidad_imagen":    {},
            "vencimiento_estado": "SIN_FECHA",
            "vencimiento_alerta": False,
            "requiere_revision": False,
        }).execute()

        doc_id = insert.data[0]["id"]
        print(f"✅ Insert OK — doc_id: {doc_id}")

        # Borrar el registro de prueba
        supabase.table("documentos").delete().eq("id", doc_id).execute()
        print(f"✅ Delete OK — registro de prueba eliminado")

    except Exception as e:
        print(f"❌ Error en insert/delete: {e}")

    print("\n🎉 Test completo")

asyncio.run(test_conexion())