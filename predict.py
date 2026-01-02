import sys
import os
import pandas as pd
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure src is in path
sys.path.append(os.path.abspath('src'))
from src.etl import cargar_datos, actualizar_datos
from src.engines import LottoEngines

def main():
    print("⏳ Actualizando base de datos histórica (esto puede tardar un poco si es la primera vez)...")
    try:
        msg = actualizar_datos()
        print(msg)
    except Exception as e:
        print(f"⚠️ Error actualizando datos: {e}")
        # Proceed if we have at least some data
    
    df = cargar_datos()
    if df.empty:
        print("❌ Error: No se pudieron descargar datos. No hay histórico para predecir.")
        return

    print(f"✅ Datos listos. Registros: {len(df)}")
    print(f"📅 Último sorteo registrado: {df.iloc[-1]['fecha']}")
    
    print("\n🧠 Generando predicciones para el Sábado...")
    engines = LottoEngines(df)
    
    print("   ... Ejecutando Ingeniero (IA)...")
    pred_ai, r_ai = engines.engine_lstm_engineer()
    
    print("   ... Ejecutando Estadístico...")
    pred_stat, r_stat = engines.engine_statistician()
    
    print("   ... Ejecutando Estratega...")
    pred_game, r_game = engines.engine_game_theory()

    print("\n========================================")
    print("🎱 RESULTADOS DE LA PREDICCIÓN CON REINTEGRO")
    print("========================================")
    print(f"🧠 IA (LSTM):       {pred_ai} | R: {r_ai}")
    print(f"📊 Estadístico:     {pred_stat} | R: {r_stat}")
    print(f"♟️ Estratega (EV+): {pred_game} | R: {r_game}")
    print("----------------------------------------")
    
    all_nums = pred_ai + pred_stat + pred_game
    consenso_series = pd.Series(all_nums).value_counts().head(6)
    consenso = sorted([int(x) for x in consenso_series.index.tolist()])
    
    all_rs = [int(r_ai), int(r_stat), int(r_game)]
    consenso_r = max(set(all_rs), key=all_rs.count) # Moda
    
    print(f"🏆 PREDICCIÓN FINAL (Consenso): {consenso} | 🔴 Reintegro: {consenso_r}")
    print("========================================")

if __name__ == "__main__":
    main()
