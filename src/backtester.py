# src/backtester.py
"""
Módulo de backtesting para LottoMind 2.0.
Mide la precisión histórica de cada engine contra sorteos reales pasados.
"""
import pandas as pd
import numpy as np
from src.engines import LottoEngines


def backtest_engine(df: pd.DataFrame, engine_name: str, n_tests: int = 50) -> dict:
    """
    Ejecuta un engine contra los últimos n_tests sorteos reales del histórico.

    Parámetros:
        df          : DataFrame completo de sorteos históricos.
        engine_name : Nombre del método en LottoEngines a evaluar.
                      Opciones: 'engine_lstm_engineer', 'engine_statistician',
                                'engine_game_theory', 'engine_markov', 'engine_decades'
        n_tests     : Cuántos sorteos históricos usar como test (máx recomendado: 100).

    Retorna:
        dict con claves:
            'matches_dist'  → lista de aciertos por test
            'avg_matches'   → media de aciertos sobre 6 posibles
            'hits'          → dict {3: count, 4: count, 5: count, 6: count}
            'engine'        → nombre del engine
            'n_tests'       → tests realizados
    """
    if len(df) < n_tests + 50:
        n_tests = max(10, len(df) // 10)

    matches_dist = []
    hits = {3: 0, 4: 0, 5: 0, 6: 0}
    errors = 0

    for i in range(n_tests, 0, -1):
        # Datos de entrenamiento: todo excepto los últimos i sorteos
        df_train = df.iloc[:-i].copy()
        real_row = df.iloc[-i]
        real_draw = set(
            int(real_row[col]) for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
        )

        try:
            engines = LottoEngines(df_train)
            engine_fn = getattr(engines, engine_name)
            pred, _ = engine_fn()
            pred_set = set(pred)
            n_matches = len(pred_set & real_draw)
            matches_dist.append(n_matches)
            if n_matches >= 3:
                hits[n_matches] = hits.get(n_matches, 0) + 1
        except Exception as e:
            errors += 1
            continue

    avg = float(np.mean(matches_dist)) if matches_dist else 0.0

    return {
        'engine': engine_name,
        'n_tests': n_tests,
        'tests_ok': len(matches_dist),
        'errors': errors,
        'matches_dist': matches_dist,
        'avg_matches': round(avg, 3),
        'hits': hits,
    }


def backtest_all_engines(df: pd.DataFrame, n_tests: int = 30, skip_slow: bool = True) -> pd.DataFrame:
    """
    Ejecuta el backtesting de todos los engines disponibles y devuelve un DataFrame comparativo.

    Parámetros:
        df         : DataFrame histórico completo.
        n_tests    : Sorteos a usar como test por engine.
        skip_slow  : Si True, omite engine_lstm_engineer y engine_prophet (lentos).

    Retorna:
        DataFrame con una fila por engine y columnas de rendimiento.
    """
    engines_to_test = [
        'engine_statistician',
        'engine_game_theory',
        'engine_markov',
        'engine_decades',
        'engine_clusters',
        'engine_genetic',
    ]

    if not skip_slow:
        engines_to_test = ['engine_lstm_engineer'] + engines_to_test + ['engine_prophet']

    results = []
    for eng in engines_to_test:
        print(f"  🔬 Backtesting {eng}...")
        result = backtest_engine(df, eng, n_tests=n_tests)
        results.append({
            'Engine': eng.replace('engine_', '').replace('_', ' ').title(),
            'Avg. Aciertos (de 6)': result['avg_matches'],
            '3 Aciertos': result['hits'].get(3, 0),
            '4 Aciertos': result['hits'].get(4, 0),
            '5 Aciertos': result['hits'].get(5, 0),
            '6 Aciertos': result['hits'].get(6, 0),
            'Tests OK': result['tests_ok'],
            'Errores': result['errors'],
        })

    return pd.DataFrame(results).sort_values('Avg. Aciertos (de 6)', ascending=False)


def get_best_engine(df: pd.DataFrame, n_tests: int = 30) -> str:
    """Devuelve el nombre del engine con mejor rendimiento en backtesting."""
    summary = backtest_all_engines(df, n_tests=n_tests, skip_slow=True)
    if summary.empty:
        return 'engine_statistician'
    best_label = summary.iloc[0]['Engine'].lower().replace(' ', '_')
    return f'engine_{best_label}'


def get_engine_weights(df: pd.DataFrame, n_tests: int = 20) -> dict:
    """
    Calcula pesos para cada engine basados en su precisión en backtesting.
    Útil para el consenso ponderado.
    """
    summary = backtest_all_engines(df, n_tests=n_tests, skip_slow=True)
    if summary.empty:
        return {}

    # Normalizar Avg. Aciertos para que sumen 1 (softmax-like o linear)
    # Usamos una versión lineal simple: (val / sum_val)
    scores = summary.set_index('Engine')['Avg. Aciertos (de 6)'].to_dict()
    total = sum(scores.values()) or 1

    weights = {k.lower().replace(' ', '_'): v / total for k, v in scores.items()}
    return weights
