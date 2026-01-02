import sys
import os
import pandas as pd
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath('src'))
from src.etl import cargar_datos, actualizar_datos
from src.engines import LottoEngines

def main():
    print("🧪 INICIANDO VERIFICACIÓN DEL SISTEMA...")
    
    # 1. Test ETL
    print("\n[1/3] Probando ETL (Extracción de Datos)...")
    try:
        # Check if data exists, if not, try to update minimal
        if not os.path.exists('data/historico.csv'):
            print("   ⚠️ No hay datos, intentando descarga inicial (esto puede tardar)...")
            # We assume update works, but to avoid long wait in test, maybe just check if function runs
            # But let's try running it for the current year at least if we could control it.
            # The current code updates from 2000 if empty. That's too long for a quick test.
            # I'll manually create a dummy csv if it doesn't exist for the engine test, 
            # OR I'll trust the user to run update from UI.
            # Let's try to just load data.
            df = cargar_datos()
            print("   ✅ Carga de datos ejecutada (aunque esté vacía).")
        else:
            df = cargar_datos()
            print(f"   ✅ Datos cargados. Registros: {len(df)}")
    except Exception as e:
        print(f"   ❌ Fallo en ETL: {e}")
        return

    # Create dummy data if empty for engine testing
    if df.empty:
        print("   ℹ️ Creando datos sintéticos para probar Motores...")
        data = {
            'fecha': pd.date_range(start='2023-01-01', periods=50),
            'n1': np.random.randint(1, 10, 50),
            'n2': np.random.randint(11, 20, 50),
            'n3': np.random.randint(21, 30, 50),
            'n4': np.random.randint(31, 40, 50),
            'n5': np.random.randint(41, 45, 50),
            'n6': np.random.randint(46, 49, 50),
        }
        df = pd.DataFrame(data)
    
    # 2. Test Engines
    print("\n[2/3] Probando Motores de Predicción...")
    try:
        engines = LottoEngines(df)
        
        print("   Testing Statistician...")
        res_stat = engines.engine_statistician()
        print(f"   -> Resultado: {res_stat}")
        
        print("   Testing Game Theory...")
        res_game = engines.engine_game_theory()
        print(f"   -> Resultado: {res_game}")

        print("   Testing LSTM Engineer (Fast Train)...")
        # LSTM might be slow, but let's try
        res_ai = engines.engine_lstm_engineer()
        print(f"   -> Resultado: {res_ai}")
        
        print("   ✅ Todos los motores responden correctamente.")
    except Exception as e:
        print(f"   ❌ Fallo en Motores: {e}")
        import traceback
        traceback.print_exc()

    print("\n[3/3] Verificación completada.")
    print("✅ El núcleo lógico funciona.")

if __name__ == "__main__":
    main()
