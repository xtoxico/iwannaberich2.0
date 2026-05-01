import sys
import os
import pandas as pd
import warnings

# Suppress minor warnings for clean output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure src is in path
sys.path.append(os.path.abspath('src'))
from src.etl import cargar_datos, actualizar_datos, proximo_sorteo, nombre_dia_sorteo
from src.engines import LottoEngines
from src.backtester import get_engine_weights


def main():
    print("⏳ Actualizando base de datos histórica...")
    try:
        msg = actualizar_datos()
        print(msg)
    except Exception as e:
        print(f"⚠️ Error actualizando datos: {e}")

    df = cargar_datos()
    if df.empty:
        print("❌ Error: No se pudieron descargar datos.")
        return

    print(f"✅ Datos listos. Registros: {len(df):,}")
    print(f"📅 Último sorteo registrado: {df.iloc[-1]['fecha'].strftime('%d/%m/%Y')}")

    proximo = proximo_sorteo()
    if proximo:
        nombre = nombre_dia_sorteo(proximo)
        print(f"🗓️  Generando predicción para el {nombre} {proximo.strftime('%d/%m/%Y')}...")

    engines = LottoEngines(df)

    print("\n   ... Ejecutando motores de IA y Estadística...")
    p_stat, r_stat = engines.engine_statistician()
    p_game, r_game = engines.engine_game_theory()
    p_markov, r_markov = engines.engine_markov()
    p_dec, r_dec = engines.engine_decades()
    p_gen, r_gen = engines.engine_genetic()
    p_clust, r_clust = engines.engine_clusters()

    print("   ... Calculando pesos adaptativos por Backtesting...")
    weights = get_engine_weights(df, n_tests=20)
    if not weights:
        weights = {k: 1.0 for k in ['statistician', 'markov', 'decades', 'game_theory', 'genetic', 'clusters']}

    print("\n========================================")
    print("🎱 RESULTADOS DE LOS MOTORES")
    print("========================================")
    print(f"📊 Estadístico:  {p_stat} | R: {r_stat} | w: {weights.get('statistician', 0):.2f}")
    print(f"♟️  Estratega:    {p_game} | R: {r_game} | w: {weights.get('game_theory', 0):.2f}")
    print(f"🔗 Markov:       {p_markov} | R: {r_markov} | w: {weights.get('markov', 0):.2f}")
    print(f"🎯 Décadas:      {p_dec} | R: {r_dec} | w: {weights.get('decades', 0):.2f}")
    print(f"🧬 Genético:     {p_gen} | R: {r_gen} | w: {weights.get('genetic', 0):.2f}")
    print(f"🗂️  Clústeres:   {p_clust} | R: {r_clust} | w: {weights.get('clusters', 0):.2f}")
    print("----------------------------------------")

    # Acumular votos ponderados
    votes = {}
    for ball in p_stat: votes[ball] = votes.get(ball, 0) + weights.get('statistician', 1)
    for ball in p_markov: votes[ball] = votes.get(ball, 0) + weights.get('markov', 1)
    for ball in p_dec: votes[ball] = votes.get(ball, 0) + weights.get('decades', 1)
    for ball in p_game: votes[ball] = votes.get(ball, 0) + weights.get('game_theory', 1)
    for ball in p_gen: votes[ball] = votes.get(ball, 0) + weights.get('genetic', 1)
    for ball in p_clust: votes[ball] = votes.get(ball, 0) + weights.get('clusters', 1)

    consenso_series = pd.Series(votes).sort_values(ascending=False).head(6)
    consenso = sorted([int(x) for x in consenso_series.index.tolist()])

    all_rs = [r_stat, r_markov, r_dec, r_game, r_gen, r_clust]
    consenso_r = max(set(all_rs), key=all_rs.count)

    print(f"🏆 PREDICCIÓN FINAL (Consenso Adaptativo): {consenso} | 🔴 Reintegro: {consenso_r}")
    print("========================================")


if __name__ == "__main__":
    main()
