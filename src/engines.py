# src/engines.py
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from collections import Counter

# TensorFlow es opcional — si no está disponible los engines sin IA siguen funcionando
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
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

# Fecha a partir de la cual el reintegro es fiable en La Primitiva
REINTEGRO_VALID_FROM = '2004-01-01'


class LottoEngines:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)  # Garantizar índices 0-based siempre
        self.ball_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']

    # ─────────────────────────────────────────────────────────────────────────
    # MÉTODO UNIFICADO: Predicción del Reintegro
    # ─────────────────────────────────────────────────────────────────────────
    def _predict_reintegro(self, context_label='general'):
        """
        Predicción unificada del reintegro que:
        1. Filtra datos pre-2004 (reintegro no existía → guardados como 0 falso)
        2. Combina frecuencia reciente + histórica + ciclos
        3. Usa muestreo ponderado (NO determinista) para variabilidad
        """
        # Filtrar solo datos con reintegro fiable
        if 'fecha' in self.df.columns:
            df_valid = self.df[pd.to_datetime(self.df['fecha']) >= REINTEGRO_VALID_FROM]
        else:
            # Fallback: usar los últimos 2000 sorteos (aprox post-2004)
            df_valid = self.df.tail(2000)

        if df_valid.empty or len(df_valid) < 10:
            return random.randint(0, 9)

        # Componente 1: Frecuencia reciente (últimos 50 sorteos) — peso 55%
        r_reciente = df_valid['r'].tail(50).value_counts()
        r_reciente = r_reciente.reindex(range(10), fill_value=0)

        # Componente 2: Frecuencia histórica (solo post-2004) — peso 25%
        r_historico = df_valid['r'].value_counts()
        r_historico = r_historico.reindex(range(10), fill_value=0)

        # Componente 3: Ciclo de "retraso" — números que llevan más tiempo sin salir — peso 20%
        r_lag_scores = pd.Series(0.0, index=range(10))
        for r_val in range(10):
            last_positions = df_valid.index[df_valid['r'] == r_val].tolist()
            if last_positions:
                lag = len(df_valid) - last_positions[-1] - 1
                r_lag_scores[r_val] = lag
            else:
                r_lag_scores[r_val] = len(df_valid)  # Máximo lag si nunca apareció

        # Normalizar cada componente a [0, 1]
        def normalize(s):
            s_min, s_max = s.min(), s.max()
            if s_max == s_min:
                return pd.Series(1.0 / len(s), index=s.index)
            return (s - s_min) / (s_max - s_min)

        score_reciente = normalize(r_reciente.astype(float))
        score_historico = normalize(r_historico.astype(float))
        score_lag = normalize(r_lag_scores)

        # Score combinado
        combined = score_reciente * 0.55 + score_historico * 0.25 + score_lag * 0.20

        # Añadir un floor mínimo para que ningún reintegro tenga probabilidad 0
        combined = combined + 0.05
        probabilities = combined / combined.sum()

        # Muestreo ponderado — seleccionar 1 valor según las probabilidades
        pred_r = int(np.random.choice(range(10), p=probabilities.values))
        return pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # HELPER: Weighted sampling de números con temperatura
    # ─────────────────────────────────────────────────────────────────────────
    def _weighted_sample(self, scores_dict, n=6, temperature=0.7):
        """
        Selecciona n números de un diccionario {número: score} usando muestreo
        ponderado con temperatura. Temperatura más alta = más aleatorio.
        """
        numbers = list(scores_dict.keys())
        scores = np.array([scores_dict[n] for n in numbers], dtype=float)

        # Aplicar temperatura (softmax-like)
        scores = scores - scores.max()  # Estabilidad numérica
        exp_scores = np.exp(scores / max(temperature, 0.01))
        probabilities = exp_scores / exp_scores.sum()

        # Muestreo sin reemplazo
        selected_indices = np.random.choice(
            len(numbers), size=min(n, len(numbers)),
            replace=False, p=probabilities
        )
        return sorted([numbers[i] for i in selected_indices])

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 1: LSTM mejorado con features enriquecidas
    # ─────────────────────────────────────────────────────────────────────────
    def engine_lstm_engineer(self, force_train=False):
        """Deep Learning: BiLSTM profundo con features enriquecidas y lookback extendido."""
        if not TF_AVAILABLE:
            nums = sorted(random.sample(range(1, 50), 6))
            return nums, self._predict_reintegro('lstm_fallback')

        lookback = 30

        # Feature engineering enriquecido: one-hot (49) + meta-features (7) = 56 dims
        data_series = []
        for idx, row in self.df[self.ball_cols].iterrows():
            # One-hot encoding de 49 bolas
            vector = np.zeros(49)
            nums_in_row = [int(v) for v in row.values]
            for val in nums_in_row:
                vector[val - 1] = 1

            # Meta-features adicionales
            ball_sum = sum(nums_in_row) / 294.0  # Normalizado (máx teórico: 44+45+46+47+48+49=279, usamos 294)
            parity = sum(1 for n in nums_in_row if n % 2 == 0) / 6.0  # Ratio de pares
            high_low = sum(1 for n in nums_in_row if n > 24) / 6.0  # Ratio altos
            spread = (max(nums_in_row) - min(nums_in_row)) / 48.0  # Rango normalizado
            # Distribución por tercios
            tercio1 = sum(1 for n in nums_in_row if n <= 16) / 6.0
            tercio2 = sum(1 for n in nums_in_row if 17 <= n <= 33) / 6.0
            tercio3 = sum(1 for n in nums_in_row if n >= 34) / 6.0

            meta = [ball_sum, parity, high_low, spread, tercio1, tercio2, tercio3]
            full_vector = np.concatenate([vector, meta])
            data_series.append(full_vector)

        feature_dim = len(data_series[0])  # 56

        # Solo usar one-hot para las labels (predecir qué bolas salen)
        label_series = []
        for _, row in self.df[self.ball_cols].iterrows():
            vector = np.zeros(49)
            for val in row.values:
                vector[int(val) - 1] = 1
            label_series.append(vector)

        X = np.array([data_series[i:i + lookback] for i in range(len(data_series) - lookback)])
        y = np.array(label_series[lookback:])

        model = None
        if os.path.exists(MODEL_PATH) and not force_train:
            try:
                model = load_model(MODEL_PATH)
                # Verificar que la forma de entrada coincide
                expected_shape = (lookback, feature_dim)
                if model.input_shape[1:] != expected_shape:
                    model = None  # Modelo incompatible, reentrenar
            except:
                model = None

        if model is None:
            model = Sequential([
                Bidirectional(LSTM(128, return_sequences=True), input_shape=(lookback, feature_dim)),
                Dropout(0.25),
                BatchNormalization(),
                Bidirectional(LSTM(64, return_sequences=False)),
                Dropout(0.25),
                Dense(128, activation='relu'),
                Dropout(0.15),
                Dense(64, activation='relu'),
                Dense(49, activation='sigmoid')
            ])
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='binary_crossentropy')

            early_stop = EarlyStopping(patience=7, restore_best_weights=True, monitor='val_loss')
            lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

            model.fit(
                X, y,
                epochs=80,
                verbose=0,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stop, lr_scheduler]
            )
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            model.save(MODEL_PATH)

        last_seq = np.array(data_series[-lookback:]).reshape(1, lookback, feature_dim)
        pred = model.predict(last_seq, verbose=0)[0]

        # Selección con algo de muestreo ponderado (no puramente argmax)
        scores = {i + 1: float(pred[i]) for i in range(49)}
        result = self._weighted_sample(scores, n=6, temperature=0.4)

        pred_r = self._predict_reintegro('lstm')
        return result, pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 2: Estadístico (frecuencia + lag + variabilidad)
    # ─────────────────────────────────────────────────────────────────────────
    def engine_statistician(self):
        """Frecuencia + Retraso con muestreo ponderado para variabilidad."""
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

        # Muestreo ponderado del top-15 en vez de tomar top-6 determinista
        top_candidates = sorted(scores, key=scores.get, reverse=True)[:15]
        candidate_scores = {n: scores[n] for n in top_candidates}
        top_balls = self._weighted_sample(candidate_scores, n=6, temperature=0.6)

        pred_r = self._predict_reintegro('statistician')
        return top_balls, pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 3: Teoría de Juegos (Anti-Humano mejorado)
    # ─────────────────────────────────────────────────────────────────────────
    def engine_game_theory(self):
        """Combinaciones diseñadas para ser únicas y evitar compartir el premio.
        Ahora incorpora datos históricos para evitar combinaciones populares."""
        # Analizar combinaciones "populares" (más elegidas por humanos)
        popular_nums = {7, 13, 14, 21, 28, 35, 42, 49}  # Múltiplos de 7
        popular_nums.update({1, 2, 3, 4, 5, 6})  # Secuencias bajas
        popular_nums.update({11, 22, 33, 44})  # Repetidos

        while True:
            nums = sorted(random.sample(range(1, 50), 6))
            if not (115 <= sum(nums) <= 185): continue
            consecutives = sum(1 for i in range(len(nums) - 1) if nums[i + 1] == nums[i] + 1)
            if consecutives > 2: continue
            low_nums = sum(1 for n in nums if n <= 31)
            if low_nums > 4: continue
            if not any(n <= 24 for n in nums) or not any(n > 24 for n in nums): continue

            # Nuevo: penalizar si muchos números son "populares"
            popular_count = sum(1 for n in nums if n in popular_nums)
            if popular_count > 2: continue

            pred_r = self._predict_reintegro('game_theory')
            return nums, pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 4: Cadenas de Markov con variabilidad
    # ─────────────────────────────────────────────────────────────────────────
    def engine_markov(self):
        """Cadenas de Markov de 1er orden con ruido y muestreo ponderado."""
        transition_matrix = np.zeros((49, 49))

        # Ponderar transiciones recientes más que antiguas
        rows = self.df[self.ball_cols].values
        n_rows = len(rows)
        for i in range(n_rows - 1):
            current_balls = [int(x) - 1 for x in rows[i]]
            next_balls = [int(x) - 1 for x in rows[i + 1]]
            # Peso exponencial: sorteos más recientes pesan más
            recency_weight = 1.0 + 2.0 * (i / n_rows)  # De 1.0 a 3.0
            for cb in current_balls:
                for nb in next_balls:
                    transition_matrix[cb][nb] += recency_weight

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums

        last_balls = [int(x) - 1 for x in self.df[self.ball_cols].iloc[-1].values]
        scores = np.zeros(49)
        for lb in last_balls:
            scores += transition_matrix[lb]

        # Reducir score de bolas del último sorteo (evitar repetición directa)
        for lb in last_balls:
            scores[lb] *= 0.3

        # Añadir ruido gaussiano para variabilidad
        noise = np.random.normal(0, scores.std() * 0.15, 49)
        scores = np.maximum(scores + noise, 0)

        # Muestreo ponderado del top-12
        score_dict = {i + 1: float(scores[i]) for i in range(49)}
        top_candidates = sorted(score_dict, key=score_dict.get, reverse=True)[:12]
        candidate_scores = {n: score_dict[n] for n in top_candidates}
        result = self._weighted_sample(candidate_scores, n=6, temperature=0.5)

        pred_r = self._predict_reintegro('markov')
        return result, pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 5: Análisis de Décadas con variabilidad
    # ─────────────────────────────────────────────────────────────────────────
    def engine_decades(self):
        """Selección basada en las décadas más frías recientemente con variabilidad."""
        decenas = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 35), (36, 42), (43, 49)]
        recientes = self.df[self.ball_cols].tail(100)
        decena_scores = {}
        for i, (lo, hi) in enumerate(decenas):
            count = recientes.apply(lambda row: sum(1 for v in row.values if lo <= v <= hi), axis=1).sum()
            decena_scores[i] = int(count)

        sorted_decenas = sorted(decena_scores.items(), key=lambda x: x[1])

        # Frecuencias recientes (no globales) para reducir repetitividad
        recientes_500 = self.df[self.ball_cols].tail(500)
        freqs_reciente = recientes_500.stack().value_counts()

        result = []
        for idx, _ in sorted_decenas:
            lo, hi = decenas[idx]
            candidates = list(range(lo, hi + 1))
            # Construir scores por candidato dentro de la década
            candidate_scores = {}
            for n in candidates:
                freq = freqs_reciente.get(n, 0)
                candidate_scores[n] = float(freq)

            # Muestrear 1 de los top-3 de la década (no siempre el mejor)
            if candidate_scores:
                selected = self._weighted_sample(candidate_scores, n=1, temperature=0.8)
                result.extend(selected)

            if len(result) >= 6:
                break

        # Si por alguna razón tenemos menos de 6, rellenar
        while len(result) < 6:
            n = random.randint(1, 49)
            if n not in result:
                result.append(n)

        pred_r = self._predict_reintegro('decades')
        return sorted(result[:6]), pred_r


    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 7: Análisis de Clústeres (K-Means) con recencia
    # ─────────────────────────────────────────────────────────────────────────
    def engine_clusters(self):
        """Identifica clústeres de sorteos similares con ponderación por recencia."""
        if KMeans is None:
            return self.engine_game_theory()

        X = self.df[self.ball_cols].values
        kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        last_cluster = clusters[-1]
        cluster_mask = clusters == last_cluster
        cluster_draws = self.df[cluster_mask]

        if cluster_draws.empty or len(cluster_draws) < 3:
            return self.engine_game_theory()

        # Ponderar frecuencias del clúster por recencia
        cluster_indices = np.where(cluster_mask)[0]
        max_idx = cluster_indices.max()
        weighted_freqs = Counter()
        for ci in cluster_indices:
            recency_weight = 1.0 + 2.0 * (ci / max_idx) if max_idx > 0 else 1.0
            for col in self.ball_cols:
                ball = int(self.df.loc[ci, col])
                weighted_freqs[ball] += recency_weight

        last_draw = set(int(v) for v in self.df[self.ball_cols].iloc[-1].values)

        # Excluir bolas del último sorteo y construir scores
        candidate_scores = {ball: score for ball, score in weighted_freqs.items() if ball not in last_draw}

        if len(candidate_scores) < 6:
            # Fallback: incluir todas
            candidate_scores = dict(weighted_freqs)

        result = self._weighted_sample(candidate_scores, n=6, temperature=0.6)

        pred_r = self._predict_reintegro('clusters')
        return result, pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 8: Algoritmo Genético mejorado
    # ─────────────────────────────────────────────────────────────────────────
    def engine_genetic(self, generations=80, pop_size=150):
        """Evoluciona una población de combinaciones con operadores mejorados."""
        # Usar frecuencias recientes (últimos 500) y globales
        freqs_global = self.df[self.ball_cols].stack().value_counts(normalize=True).to_dict()
        freqs_recent = self.df[self.ball_cols].tail(500).stack().value_counts(normalize=True).to_dict()

        # Score de lag por número
        total = len(self.df)
        lag_scores = {}
        for n in range(1, 50):
            mask = self.df[self.ball_cols].isin([n]).any(axis=1)
            positions = self.df[mask].index.tolist()
            lag = total - positions[-1] - 1 if positions else total
            lag_scores[n] = lag / total

        def calculate_fitness(comb):
            # Frecuencia: mix de global y reciente
            f_global = sum(freqs_global.get(n, 0) for n in comb)
            f_recent = sum(freqs_recent.get(n, 0) for n in comb)
            f_score = f_global * 0.4 + f_recent * 0.6

            # Bonus por lag (números "calientes" por retraso)
            lag_bonus = sum(lag_scores.get(n, 0) for n in comb) * 0.3

            # Penalización por suma fuera de rango
            s = sum(comb)
            s_penalty = 0 if 115 <= s <= 185 else -0.8

            # Penalización por consecutivos excesivos
            c = sum(1 for i in range(len(comb) - 1) if comb[i + 1] == comb[i] + 1)
            c_penalty = -0.3 * max(c - 1, 0)

            # Bonus por diversidad de décadas
            decades_used = len(set((n - 1) // 7 for n in comb))
            diversity_bonus = decades_used * 0.05

            return f_score + lag_bonus + s_penalty + c_penalty + diversity_bonus

        population = [sorted(random.sample(range(1, 50), 6)) for _ in range(pop_size)]

        for gen in range(generations):
            population = sorted(population, key=calculate_fitness, reverse=True)
            # Elitismo: mantener el top 20%
            elite_size = max(pop_size // 5, 2)
            next_gen = population[:elite_size]

            while len(next_gen) < pop_size:
                # Selección por torneo (más variedad que coger siempre el top-10)
                tournament = random.sample(next_gen[:max(elite_size, 10)], min(3, len(next_gen)))
                parent1 = max(tournament, key=calculate_fitness)
                tournament = random.sample(next_gen[:max(elite_size, 10)], min(3, len(next_gen)))
                parent2 = max(tournament, key=calculate_fitness)

                # Crossover de 2 puntos
                cp1, cp2 = sorted(random.sample(range(6), 2))
                child_set = set(parent1[:cp1] + parent2[cp1:cp2] + parent1[cp2:])

                # Asegurar exactamente 6 números únicos
                child = list(child_set)
                while len(child) < 6:
                    n = random.randint(1, 49)
                    if n not in child:
                        child.append(n)
                child = sorted(child[:6])

                # Mutación adaptativa (más alta al principio, menos al final)
                mutation_rate = 0.15 * (1 - gen / generations) + 0.03
                if random.random() < mutation_rate:
                    idx = random.randint(0, 5)
                    new_n = random.randint(1, 49)
                    if new_n not in child:
                        child[idx] = new_n
                        child = sorted(child)

                next_gen.append(child)

            population = next_gen

        # No tomar siempre el #1; muestrear del top-5
        top5 = population[:5]
        best_comb = random.choice(top5)

        pred_r = self._predict_reintegro('genetic')
        return sorted(best_comb), pred_r

    # ─────────────────────────────────────────────────────────────────────────
    # ENGINE 9: Patrones Temporales (NUEVO)
    # ─────────────────────────────────────────────────────────────────────────
    def engine_temporal_patterns(self):
        """Analiza ciclos de aparición, co-ocurrencias y patrones por día de sorteo."""
        total = len(self.df)
        if total < 100:
            return sorted(random.sample(range(1, 50), 6)), self._predict_reintegro('temporal')

        # ── 1. Análisis de ciclo medio por número ──
        cycle_scores = {}
        for n in range(1, 50):
            mask = self.df[self.ball_cols].isin([n]).any(axis=1)
            positions = self.df[mask].index.tolist()
            if len(positions) >= 2:
                # Calcular gaps entre apariciones
                gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
                avg_cycle = np.mean(gaps)
                current_gap = total - positions[-1] - 1

                # Score: cuánto más "atrasado" está respecto a su ciclo, mayor probabilidad
                if avg_cycle > 0:
                    cycle_scores[n] = current_gap / avg_cycle
                else:
                    cycle_scores[n] = 0
            else:
                cycle_scores[n] = 2.0  # Alto score si apenas ha aparecido

        # ── 2. Co-ocurrencias (qué números tienden a salir juntos) ──
        # Usar los últimos 5 sorteos como "contexto"
        recent_balls = set()
        for i in range(-1, max(-6, -total), -1):
            for col in self.ball_cols:
                recent_balls.add(int(self.df[col].iloc[i]))

        cooccurrence_boost = Counter()
        for _, row in self.df[self.ball_cols].tail(500).iterrows():
            draw = set(int(v) for v in row.values)
            overlap = draw & recent_balls
            if len(overlap) >= 2:
                for n in draw - recent_balls:
                    cooccurrence_boost[n] += len(overlap) * 0.1

        # ── 3. Patrones por día de la semana (L/J/S) ──
        day_scores = {}
        if 'fecha' in self.df.columns:
            self.df['_dow'] = pd.to_datetime(self.df['fecha']).dt.dayofweek
            # Próximo sorteo: determinar qué día es
            from src.etl import proximo_sorteo
            proximo = proximo_sorteo()
            target_dow = proximo.weekday() if proximo else None

            if target_dow is not None:
                same_day_draws = self.df[self.df['_dow'] == target_dow]
                if len(same_day_draws) > 20:
                    day_freqs = same_day_draws[self.ball_cols].tail(200).stack().value_counts()
                    day_freqs = day_freqs.reindex(range(1, 50), fill_value=0)
                    max_day_freq = day_freqs.max() if day_freqs.max() > 0 else 1
                    for n in range(1, 50):
                        day_scores[n] = float(day_freqs.get(n, 0)) / max_day_freq

            # Limpiar columna temporal
            self.df.drop(columns=['_dow'], inplace=True, errors='ignore')

        # ── Combinar los 3 componentes ──
        combined_scores = {}
        max_cycle = max(cycle_scores.values()) if cycle_scores else 1
        max_cooc = max(cooccurrence_boost.values()) if cooccurrence_boost else 1

        for n in range(1, 50):
            s_cycle = cycle_scores.get(n, 0) / max_cycle if max_cycle > 0 else 0
            s_cooc = cooccurrence_boost.get(n, 0) / max_cooc if max_cooc > 0 else 0
            s_day = day_scores.get(n, 0.5)  # 0.5 neutral si no hay datos

            combined_scores[n] = s_cycle * 0.45 + s_cooc * 0.25 + s_day * 0.30

        result = self._weighted_sample(combined_scores, n=6, temperature=0.6)
        pred_r = self._predict_reintegro('temporal')
        return result, pred_r
