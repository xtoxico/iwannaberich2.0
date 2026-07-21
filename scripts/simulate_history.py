import sys
import os
import pandas as pd
import argparse
import time

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.etl import cargar_datos
from src.engines import LottoEngines

RESULTS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'simulation_results.csv'))

def parse_args():
    parser = argparse.ArgumentParser(description="Simulador Walk-Forward de LottoMind")
    parser.add_argument('--start-index', type=int, default=50, 
                        help='Índice del sorteo desde el que empezar a predecir (requiere un histórico previo). Por defecto: 50.')
    parser.add_argument('--end-index', type=int, default=None, 
                        help='Índice del sorteo en el que terminar. Si no se especifica, procesa hasta el final.')
    parser.add_argument('--include-slow', action='store_true', 
                        help='Incluir motores lentos (LSTM, Genético).')
    return parser.parse_args()

def main():
    args = parse_args()
    print("⏳ Cargando datos históricos...")
    df = cargar_datos()
    if df.empty:
        print("❌ Error: No hay datos históricos. Ejecuta primero predict.py o descarga los datos.")
        return

    total_draws = len(df)
    print(f"✅ Histórico cargado. Total de sorteos: {total_draws}")

    engines_to_test = [
        'engine_statistician',
        'engine_game_theory',
        'engine_markov',
        'engine_decades',
        'engine_clusters',
        'engine_temporal_patterns',
    ]

    if args.include_slow:
        engines_to_test.extend(['engine_lstm_engineer', 'engine_genetic'])

    print(f"Motores a evaluar: {', '.join([e.replace('engine_', '') for e in engines_to_test])}")

    # Cargar resultados anteriores para poder reanudar
    processed_indices = set()
    if os.path.exists(RESULTS_FILE):
        try:
            results_df = pd.read_csv(RESULTS_FILE)
            if not results_df.empty and 'draw_index' in results_df.columns:
                processed_indices = set(results_df['draw_index'].unique())
                print(f"🔁 Se encontraron resultados anteriores. Retomando simulación (Sorteos ya procesados: {len(processed_indices)}).")
        except Exception as e:
            print(f"⚠️ Error leyendo {RESULTS_FILE}: {e}")

    start_idx = max(args.start_index, 10)  # Mínimo 10 para que los motores no fallen por falta absoluta de datos
    end_idx = args.end_index if args.end_index is not None else total_draws

    # Inicializar el archivo CSV si no existe
    if not os.path.exists(RESULTS_FILE):
        columns = ['draw_index', 'draw_date', 'engine', 'real_n1', 'real_n2', 'real_n3', 'real_n4', 'real_n5', 'real_n6', 'real_r', 
                   'pred_n1', 'pred_n2', 'pred_n3', 'pred_n4', 'pred_n5', 'pred_n6', 'pred_r', 'matches']
        pd.DataFrame(columns=columns).to_csv(RESULTS_FILE, index=False)

    print("\n🚀 Iniciando Simulación Walk-Forward...")
    
    for i in range(start_idx, end_idx):
        if i in processed_indices:
            continue

        real_row = df.iloc[i]
        real_draw = set(int(real_row[col]) for col in ['n1', 'n2', 'n3', 'n4', 'n5', 'n6'])
        real_r = int(real_row['r'])
        draw_date = real_row['fecha'].strftime('%Y-%m-%d') if pd.notnull(real_row['fecha']) else ''

        # Cortar el dataframe hasta el momento justo antes de este sorteo
        df_train = df.iloc[:i].copy()
        
        engines = LottoEngines(df_train)
        
        print(f"Evaluando Sorteo {i}/{total_draws-1} ({draw_date}) con {len(df_train)} histórico...", end='', flush=True)
        start_time = time.time()
        
        draw_results = []
        for engine_name in engines_to_test:
            try:
                engine_fn = getattr(engines, engine_name)
                # handle args for genetic if possible, but it has defaults
                if engine_name == 'engine_lstm_engineer':
                     pred_nums, pred_r = engine_fn(force_train=False) # Don't retrain in every step!
                else:
                     pred_nums, pred_r = engine_fn()
                
                # Calcular aciertos
                pred_set = set(pred_nums)
                matches = len(pred_set & real_draw)
                
                sorted_pred = sorted(pred_nums)
                while len(sorted_pred) < 6:
                    sorted_pred.append(-1)
                
                draw_results.append({
                    'draw_index': i,
                    'draw_date': draw_date,
                    'engine': engine_name.replace('engine_', ''),
                    'real_n1': int(real_row['n1']), 'real_n2': int(real_row['n2']), 'real_n3': int(real_row['n3']),
                    'real_n4': int(real_row['n4']), 'real_n5': int(real_row['n5']), 'real_n6': int(real_row['n6']),
                    'real_r': real_r,
                    'pred_n1': sorted_pred[0], 'pred_n2': sorted_pred[1], 'pred_n3': sorted_pred[2],
                    'pred_n4': sorted_pred[3], 'pred_n5': sorted_pred[4], 'pred_n6': sorted_pred[5],
                    'pred_r': pred_r,
                    'matches': matches
                })
            except Exception as e:
                # Silencioso en errores de motores, para no parar la simulación entera
                pass

        # === ADD CONSENSUS LOGIC TO WALK-FORWARD ===
        if draw_results:
            import numpy as np
            import random
            
            # Recolectar todas las predicciones de los motores en este paso
            votes = {}
            r_votes = {}
            for res in draw_results:
                # Damos el mismo peso a todos en la simulación rápida para no hacer backtest anidado (muy lento)
                w = 1.0 
                for i in range(1, 7):
                    n = res.get(f'pred_n{i}')
                    if n and n != -1:
                        votes[n] = votes.get(n, 0) + w
                
                r = res.get('pred_r')
                if r is not None and r != -1:
                     r_votes[r] = r_votes.get(r, 0) + w
            
            consenso_series = pd.Series(votes).sort_values(ascending=False)
            
            # Reintegro por consenso
            consenso_r = -1
            if r_votes:
                r_series = pd.Series(r_votes)
                r_probs = r_series / r_series.sum()
                consenso_r = int(np.random.choice(r_probs.index, p=r_probs.values))

            # 1. Conservadora
            conservadora = sorted([int(x) for x in consenso_series.head(6).index.tolist()])
            
            # 2. Equilibrada
            top3 = [int(x) for x in consenso_series.head(3).index.tolist()]
            remaining = {int(x): float(consenso_series[x]) for x in consenso_series.index[3:12] if int(x) not in top3}
            if remaining:
                remaining_keys = list(remaining.keys())
                remaining_vals = np.array(list(remaining.values()))
                remaining_probs = remaining_vals / remaining_vals.sum()
                extra_3 = list(np.random.choice(remaining_keys, size=min(3, len(remaining_keys)), replace=False, p=remaining_probs[:len(remaining_keys)]))
            else:
                extra_3 = sorted(random.sample(range(1, 50), 3))
            equilibrada = sorted(top3 + [int(x) for x in extra_3])

            # 3. Agresiva
            top2 = [int(x) for x in consenso_series.head(2).index.tolist()]
            tail_candidates = {int(x): float(consenso_series[x]) for x in consenso_series.index[6:] if int(x) not in top2}
            if len(tail_candidates) >= 4:
                tail_keys = list(tail_candidates.keys())
                tail_vals = np.array(list(tail_candidates.values()))
                tail_probs = tail_vals / tail_vals.sum()
                extra_4 = list(np.random.choice(tail_keys, size=4, replace=False, p=tail_probs[:len(tail_keys)]))
            else:
                extra_4 = sorted(random.sample([n for n in range(1, 50) if n not in top2], 4))
            agresiva = sorted(top2 + [int(x) for x in extra_4])

            # Evaluar los 3 consensos
            for name, pred_nums in [('consensus_conservative', conservadora), 
                                    ('consensus_balanced', equilibrada), 
                                    ('consensus_aggressive', agresiva)]:
                pred_set = set(pred_nums)
                matches = len(pred_set & real_draw)
                sorted_pred = sorted(pred_nums)
                while len(sorted_pred) < 6:
                    sorted_pred.append(-1)
                
                draw_results.append({
                    'draw_index': i,
                    'draw_date': draw_date,
                    'engine': name,
                    'real_n1': int(real_row['n1']), 'real_n2': int(real_row['n2']), 'real_n3': int(real_row['n3']),
                    'real_n4': int(real_row['n4']), 'real_n5': int(real_row['n5']), 'real_n6': int(real_row['n6']),
                    'real_r': real_r,
                    'pred_n1': sorted_pred[0], 'pred_n2': sorted_pred[1], 'pred_n3': sorted_pred[2],
                    'pred_n4': sorted_pred[3], 'pred_n5': sorted_pred[4], 'pred_n6': sorted_pred[5],
                    'pred_r': consenso_r,
                    'matches': matches
                })


        if draw_results:
            # Guardar iterativamente (append)
            res_df = pd.DataFrame(draw_results)
            res_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
            
        elapsed = time.time() - start_time
        print(f" Hecho. ({elapsed:.2f}s)")

    print(f"\n✅ Simulación completada. Resultados guardados en {RESULTS_FILE}")

if __name__ == "__main__":
    main()
