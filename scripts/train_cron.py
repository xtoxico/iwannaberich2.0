import sys
import os
from datetime import datetime
import csv

# Añadir raíz del proyecto al path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.etl import actualizar_datos, cargar_datos
from src.engines import LottoEngines, PREDICTIONS_PATH

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
        print("🧠 Re-entrenando red neuronal (BiLSTM)...")
        # Forzar entrenamiento para actualizar pesos con datos nuevos
        nums, r = engines.engine_lstm_engineer(force_train=True)
        print(f"✅ Modelo guardado en models/lotto_lstm.keras")
        print(f"🔮 Predicción LSTM: {nums} R:{r}")
        
        # 3. Guardar predicciones en predictions.csv
        print("💾 Guardando predicciones...")
        save_prediction(nums, r, 'lstm')

        # Generar y guardar predicciones de otros engines
        for engine_name, method in [
            ('statistician', engines.engine_statistician),
            ('markov', engines.engine_markov),
            ('temporal', engines.engine_temporal_patterns),
            ('genetic', engines.engine_genetic),
        ]:
            try:
                pred_nums, pred_r = method()
                save_prediction(pred_nums, pred_r, engine_name)
                print(f"  ✅ {engine_name}: {pred_nums} R:{pred_r}")
            except Exception as e:
                print(f"  ⚠️ Error en {engine_name}: {e}")

        print("✨ Tarea de Cron finalizada con éxito.")
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")


def save_prediction(nums, reintegro, engine_name):
    """Guarda una predicción en predictions.csv con timestamp."""
    os.makedirs(os.path.dirname(PREDICTIONS_PATH), exist_ok=True)
    
    file_exists = os.path.exists(PREDICTIONS_PATH) and os.path.getsize(PREDICTIONS_PATH) > 0
    
    with open(PREDICTIONS_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['fecha', 'engine', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'r'])
        
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            engine_name,
            *nums,
            reintegro
        ])


if __name__ == "__main__":
    main()
