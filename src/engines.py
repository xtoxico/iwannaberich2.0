# src/engines.py
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# TensorFlow es opcional — si no está disponible los engines sin IA siguen funcionando
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lotto_lstm.keras')
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'data', 'predictions.csv')

class LottoEngines:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)  # Garantizar índices 0-based siempre
        self.ball_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 1: LSTM mejorado
    # ─────────────────────────────────────────────────────────────────────────
    def engine_lstm_engineer(self, force_train=False):
        """Deep Learning: BiLSTM profundo con lookback extendido y todos los datos."""
        if not TF_AVAILABLE:
            # Fallback a un generador pseudo-aleatorio inteligente si no hay TF
            nums = sorted(random.sample(range(1, 50), 6))
            return nums, random.randint(0, 9)

        lookback = 30  # Aumentado de 10 → 30 sorteos de contexto

        data_series = []
        for _, row in self.df[self.ball_cols].iterrows():
            vector = np.zeros(49)
            for val in row.values:
                vector[int(val) - 1] = 1
            data_series.append(vector)

        X = np.array([data_series[i:i + lookback] for i in range(len(data_series) - lookback)])
        y = np.array(data_series[lookback:])  # Etiquetas desplazadas correctamente

        model = None
        if os.path.exists(MODEL_PATH) and not force_train:
            try:
                model = load_model(MODEL_PATH)
            except:
                model = None

        if model is None:
            # Arquitectura más profunda: 2 capas BiLSTM + Dropout + Dense intermedia
            model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True), input_shape=(lookback, 49)),
                Dropout(0.2),
                Bidirectional(LSTM(64, return_sequences=False)),
                Dropout(0.2),
                Dense(128, activation='relu'),
                Dense(49, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy')

            # Entrenar con TODOS los datos disponibles
            early_stop = EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
            model.fit(
                X, y,
                epochs=50,
                verbose=0,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stop]
            )
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            model.save(MODEL_PATH)

        last_seq = np.array(data_series[-lookback:]).reshape(1, lookback, 49)
        pred = model.predict(last_seq, verbose=0)[0]
        top_indices = pred.argsort()[-6:][::-1]

        # Reintegro: tendencia reciente ponderada (70%) + histórico (30%)
        r_reciente = self.df['r'].tail(20).value_counts()
        r_historico = self.df['r'].value_counts()
        r_score = r_reciente * 0.7 + r_historico.reindex(r_reciente.index, fill_value=0) * 0.3
        pred_r = int(r_score.idxmax()) if not r_score.empty else 0

        return [int(i + 1) for i in sorted(top_indices)], pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 2: Estadístico (frecuencia + lag corregido)
    # ─────────────────────────────────────────────────────────────────────────
    def engine_statistician(self):
        """Frecuencia + Retraso (lag calculado con posiciones ordinales correctas)."""
        df_reset = self.df 

        freqs = df_reset[self.ball_cols].stack().value_counts().sort_index()
        freqs = freqs.reindex(range(1, 50), fill_value=0)

        total = len(df_reset)
        last_seen = {}
        for n in range(1, 50):
            mask = df_reset[self.ball_cols].isin([n]).any(axis=1)
            positions = df_reset[mask].index.tolist()
            last_seen[n] = total - positions[-1] - 1 if positions else total

        max_freq = freqs.max() if freqs.max() > 0 else 1
        max_lag = max(last_seen.values()) if max(last_seen.values()) > 0 else 1

        scores = {}
        for n in range(1, 50):
            norm_freq = freqs[n] / max_freq
            norm_lag = last_seen[n] / max_lag
            scores[n] = (norm_lag * 0.6) + (norm_freq * 0.4)

        top_balls = sorted(sorted(scores, key=scores.get, reverse=True)[:6])

        r_counts = self.df['r'].value_counts()
        most_common_r = int(r_counts.idxmax()) if not r_counts.empty else 0

        return top_balls, most_common_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 3: Teoría de Juegos (Anti-Humano)
    # ─────────────────────────────────────────────────────────────────────────
    def engine_game_theory(self):
        """Combinaciones diseñadas para ser únicas y evitar compartir el premio."""
        while True:
            nums = sorted(random.sample(range(1, 50), 6))
            if not (115 <= sum(nums) <= 185): continue
            consecutives = sum(1 for i in range(len(nums) - 1) if nums[i + 1] == nums[i] + 1)
            if consecutives > 2: continue
            low_nums = sum(1 for n in nums if n <= 31)
            if low_nums > 4: continue
            if not any(n <= 24 for n in nums) or not any(n > 24 for n in nums): continue
            
            r_val = random.randint(0, 9)
            return nums, r_val

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 4: Cadenas de Markov
    # ─────────────────────────────────────────────────────────────────────────
    def engine_markov(self):
        """Cadenas de Markov de 1er orden: probabilidad de transición entre sorteos."""
        transition_matrix = np.zeros((49, 49))
        rows = self.df[self.ball_cols].values
        for i in range(len(rows) - 1):
            current_balls = [int(x) - 1 for x in rows[i]]
            next_balls = [int(x) - 1 for x in rows[i + 1]]
            for cb in current_balls:
                for nb in next_balls:
                    transition_matrix[cb][nb] += 1

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums

        last_balls = [int(x) - 1 for x in self.df[self.ball_cols].iloc[-1].values]
        scores = np.zeros(49)
        for lb in last_balls:
            scores += transition_matrix[lb]
        scores[last_balls] = 0

        top_indices = scores.argsort()[-6:][::-1]
        pred_r = int(self.df['r'].value_counts().idxmax()) if not self.df['r'].empty else 0
        return [int(i + 1) for i in sorted(top_indices)], pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 5: Análisis de Décadas
    # ─────────────────────────────────────────────────────────────────────────
    def engine_decades(self):
        """Selección basada en las décadas más frías recientemente."""
        decenas = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42), (43, 49)]
        recientes = self.df[self.ball_cols].tail(100)
        decena_scores = {}
        for i, (lo, hi) in enumerate(decenas):
            count = recientes.apply(lambda row: sum(1 for v in row.values if lo <= v <= hi), axis=1).sum()
            decena_scores[i] = int(count)

        sorted_decenas = sorted(decena_scores.items(), key=lambda x: x[1])
        freqs_global = self.df[self.ball_cols].stack().value_counts()
        result = []
        for idx, _ in sorted_decenas:
            lo, hi = decenas[idx]
            candidates = list(range(lo, hi + 1))
            best = max(candidates, key=lambda n: freqs_global.get(n, 0))
            result.append(best)
            if len(result) == 6: break
        pred_r = int(self.df['r'].value_counts().idxmax()) if not self.df['r'].empty else 0
        return sorted(result), pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 6: Prophet
    # ─────────────────────────────────────────────────────────────────────────
    def engine_prophet(self, target_date=None):
        """Predicción con Facebook Prophet."""
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet no está instalado.")

        if target_date is None:
            from src.etl import proximo_sorteo
            target_date = proximo_sorteo()

        ball_scores = {}
        df_sub = self.df.tail(800).copy()
        for ball in range(1, 50):
            ts = df_sub[['fecha']].copy().rename(columns={'fecha': 'ds'})
            ts['y'] = df_sub[self.ball_cols].apply(lambda row: 1.0 if ball in row.values else 0.0, axis=1).values
            ts['ds'] = pd.to_datetime(ts['ds']).dt.tz_localize(None)
            m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, changepoint_prior_scale=0.05, seasonality_mode='additive')
            m.fit(ts, iter=300)
            future = pd.DataFrame({'ds': [pd.Timestamp(target_date).tz_localize(None)]})
            forecast = m.predict(future)
            ball_scores[ball] = float(forecast['yhat'].values[0])

        top_balls = sorted(sorted(ball_scores, key=ball_scores.get, reverse=True)[:6])
        pred_r = int(self.df['r'].value_counts().idxmax()) if not self.df['r'].empty else 0
        return top_balls, pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 7: Análisis de Clústeres (K-Means)
    # ─────────────────────────────────────────────────────────────────────────
    def engine_clusters(self):
        """Identifica clústeres de sorteos similares."""
        if KMeans is None: return self.engine_game_theory()
        X = self.df[self.ball_cols].values
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        last_cluster = clusters[-1]
        cluster_draws = self.df[clusters == last_cluster]
        cluster_freqs = cluster_draws[self.ball_cols].stack().value_counts()
        last_draw = set(self.df[self.ball_cols].iloc[-1].values)
        top_balls = []
        for ball in cluster_freqs.index:
            if ball not in last_draw: top_balls.append(int(ball))
            if len(top_balls) == 6: break
        pred_r = int(cluster_draws['r'].value_counts().idxmax()) if not cluster_draws['r'].empty else 0
        return sorted(top_balls), pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 8: Algoritmo Genético
    # ─────────────────────────────────────────────────────────────────────────
    def engine_genetic(self, generations=50, pop_size=100):
        """Evoluciona una población de combinaciones."""
        freqs = self.df[self.ball_cols].stack().value_counts(normalize=True).to_dict()
        def calculate_fitness(comb):
            f_score = sum(freqs.get(n, 0) for n in comb)
            s = sum(comb)
            s_penalty = 0 if 115 <= s <= 185 else -0.5
            c = sum(1 for i in range(len(comb)-1) if comb[i+1] == comb[i]+1)
            c_penalty = -0.2 * c
            return f_score + s_penalty + c_penalty

        population = [sorted(random.sample(range(1, 50), 6)) for _ in range(pop_size)]
        for _ in range(generations):
            population = sorted(population, key=calculate_fitness, reverse=True)
            next_gen = population[:pop_size // 5]
            while len(next_gen) < pop_size:
                parent1, parent2 = random.sample(next_gen[:10], 2)
                child = sorted(list(set(parent1[:3] + parent2[3:])))
                while len(child) < 6:
                    n = random.randint(1, 49)
                    if n not in child: child.append(n)
                child = sorted(child[:6])
                if random.random() < 0.05:
                    idx = random.randint(0, 5)
                    new_n = random.randint(1, 49)
                    if new_n not in child: child[idx] = new_n
                next_gen.append(sorted(child))
            population = next_gen
        best_comb = population[0]
        pred_r = int(self.df['r'].value_counts().idxmax()) if not self.df['r'].empty else 0
        return best_comb, pred_r
