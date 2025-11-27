from langfuse import observe, Langfuse
import os
from dotenv import load_dotenv

def test_final():
    load_dotenv()
    
    # Asegurar que las variables de entorno están disponibles para el decorador
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("❌ Faltan claves en .env")
        return

    print("🚀 Probando observabilidad con decorador @observe...")
    
    @observe(name="prueba_conectividad")
    def funcion_test():
        print("   ✅ Función ejecutada bajo observación.")
        return "éxito"

    try:
        funcion_test()
        
        # Forzar flush para asegurar envío inmediato
        lf = Langfuse()
        lf.flush()
        print("\n✅ Traza enviada correctamente.")
        print("👉 Revisa tu dashboard en Langfuse para confirmar.")
    except Exception as e:
        print(f"\n❌ Error al enviar traza: {e}")

if __name__ == "__main__":
    test_final()
