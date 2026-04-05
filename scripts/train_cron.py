import sys
import os
from datetime import datetime

# Añadir raíz del proyecto al path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.etl import actualizar_datos, cargar_datos
from src.engines import LottoEngines

def main():
    print(f"🔄 [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Iniciando Cron: Actualización y Entrenamiento")
    
    # 1. Actualizar Datos
    try:
        msg = actualizar_datos()
        print(f"📊 {msg}")
    except Exception as e:
        print(f"❌ Error actualizando datos: {e}")
        return

    # 2. Cargar Datos y Entrenar IA
    try:
        df = cargar_datos()
        if df.empty:
            print("❌ Error: No hay datos para entrenar.")
            return

        engines = LottoEngines(df)
        print("🧠 Re-entrenando red neuronal (LSTM)...")
        # Forzar entrenamiento para actualizar pesos con datos nuevos
        nums, r = engines.engine_lstm_engineer(force_train=True)
        print(f"✅ Modelo guardado en models/lotto_lstm.keras")
        print(f"🔮 Predicción sugerida para el próximo sorteo: {nums} R:{r}")
        
        # 3. Pre-bloquear predicciones
        print("💾 Guardando predicción oficial para evitar cambios...")
        engines.get_locked_prediction('IA')
        engines.get_locked_prediction('Estadistico')
        engines.get_locked_prediction('Estratega')
        print("✨ Tarea de Cron finalizada con éxito.")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")

if __name__ == "__main__":
    main()
