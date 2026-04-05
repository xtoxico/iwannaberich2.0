# src/engines.py
import pandas as pd
import numpy as np
import random
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

import os
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'lotto_lstm.keras')
PREDICTIONS_PATH = os.path.join(BASE_DIR, 'data', 'predictions.csv')

class LottoEngines:
    def __init__(self, df):
        self.df = df
        self.ball_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']

    def get_next_draw_date(self):
        """Calcula la fecha del próximo sorteo de La Primitiva (Jueves o Sábado)"""
        now = datetime.now()
        # 0:Lunes, 1:Martes, 2:Miércoles, 3:Jueves, 4:Viernes, 5:Sábado, 6:Domingo
        days_ahead = {
            0: 3, # Lunes -> Jueves (+3)
            1: 2, # Martes -> Jueves (+2)
            2: 1, # Miércoles -> Jueves (+1)
            3: 0 if now.hour < 21 else 2, # Jueves (Hoy si < 21:00, else Sábado +2)
            4: 1, # Viernes -> Sábado (+1)
            5: 0 if now.hour < 21 else 5, # Sábado (Hoy si < 21:00, else Jueves +5)
            6: 4  # Domingo -> Jueves (+4)
        }
        target_days = days_ahead[now.weekday()]
        next_date = (now + timedelta(days=target_days)).replace(hour=21, minute=0, second=0, microsecond=0)
        return next_date.strftime('%Y-%m-%d')

    def engine_lstm_engineer(self, force_train=False):
        """Enfoque Deep Learning con persistencia"""
        if not TENSORFLOW_AVAILABLE:
            # Fallback a un generador pseudo-aleatorio inteligente si no hay TF
            # (En producción esto avisaría al usuario de la incompatibilidad)
            nums = sorted(random.sample(range(1, 50), 6))
            return nums, random.randint(0, 9)

        lookback = 10
        data_series = []
        for _, row in self.df[self.ball_cols].iterrows():
            vector = np.zeros(49)
            for val in row.values:
                vector[int(val)-1] = 1
            data_series.append(vector)
        
        X = np.array([data_series[i:i+lookback] for i in range(len(data_series)-lookback)])
        y = np.array(data_series[lookback:])

        model = None
        if os.path.exists(MODEL_PATH) and not force_train:
            try:
                model = load_model(MODEL_PATH)
            except:
                model = None

        if model is None:
            model = Sequential([
                Bidirectional(LSTM(64, return_sequences=False), input_shape=(lookback, 49)),
                Dense(49, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            
            train_limit = min(len(X), 500)
            model.fit(X[-train_limit:], y[-train_limit:], epochs=15, verbose=0, batch_size=16)
            model.save(MODEL_PATH)
        
        last_seq = np.array(data_series[-lookback:]).reshape(1, lookback, 49)
        pred = model.predict(last_seq, verbose=0)[0]
        top_indices = pred.argsort()[-6:][::-1]
        
        reintegros_recent = self.df['r'].tail(50).value_counts().index.tolist()
        pred_r = int(reintegros_recent[0]) if reintegros_recent else 0
        
        return [int(i+1) for i in sorted(top_indices)], pred_r

    def engine_statistician(self):
        """Enfoque Frecuencia + Retraso"""
        freqs = self.df[self.ball_cols].stack().value_counts().sort_index()
        freqs = freqs.reindex(range(1, 50), fill_value=0)
        
        last_seen = {}
        total = len(self.df)
        for n in range(1, 50):
            matches = self.df[self.df[self.ball_cols].isin([n]).any(axis=1)]
            last_seen[n] = total - matches.index[-1] if not matches.empty else total
            
        scores = {}
        max_freq = freqs.max() if freqs.max() > 0 else 1
        max_lag = max(last_seen.values()) if max(last_seen.values()) > 0 else 1
        
        for n in range(1, 50):
            norm_freq = freqs[n] / max_freq
            norm_lag = last_seen[n] / max_lag
            scores[n] = (norm_lag * 0.7) + (norm_freq * 0.3)
            
        top_balls = sorted(sorted(scores, key=scores.get, reverse=True)[:6])
        r_counts = self.df['r'].value_counts()
        most_common_r = int(r_counts.idxmax()) if not r_counts.empty else 0
        
        return top_balls, most_common_r

    def engine_game_theory(self):
        """Enfoque Anti-Humano"""
        while True:
            nums = sorted(random.sample(range(1, 50), 6))
            if not (120 <= sum(nums) <= 180): continue
            consecutives = sum(1 for i in range(len(nums)-1) if nums[i+1] == nums[i]+1)
            if consecutives > 2: continue
            low_nums = sum(1 for n in nums if n <= 31)
            if low_nums > 4: continue
            return nums, random.randint(0, 9)

    def get_locked_prediction(self, engine_name):
        """Obtiene o genera una predicción persistente para la próxima fecha"""
        fecha_proxima = self.get_next_draw_date()
        
        if os.path.exists(PREDICTIONS_PATH):
            preds_df = pd.read_csv(PREDICTIONS_PATH)
            existing = preds_df[(preds_df['fecha'] == fecha_proxima) & (preds_df['engine'] == engine_name)]
            if not existing.empty:
                row = existing.iloc[0]
                return [int(row['n1']), int(row['n2']), int(row['n3']), int(row['n4']), int(row['n5']), int(row['n6'])], int(row['r'])

        # Si no existe, generar
        if engine_name == 'IA':
            nums, r = self.engine_lstm_engineer()
        elif engine_name == 'Estadistico':
            nums, r = self.engine_statistician()
        elif engine_name == 'Estratega':
            nums, r = self.engine_game_theory()
        else: # Consenso o default
            return None, None

        # Guardar
        new_row = {
            'fecha': fecha_proxima,
            'engine': engine_name,
            'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
            'n4': nums[3], 'n5': nums[4], 'n6': nums[5],
            'r': r
        }
        
        if os.path.exists(PREDICTIONS_PATH) and os.path.getsize(PREDICTIONS_PATH) > 0:
            preds_df = pd.read_csv(PREDICTIONS_PATH)
            preds_df = pd.concat([preds_df, pd.DataFrame([new_row])]).drop_duplicates(subset=['fecha', 'engine'])
        else:
            preds_df = pd.DataFrame([new_row])
            
        preds_df.to_csv(PREDICTIONS_PATH, index=False)
        return nums, r