# src/engines.py
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional

class LottoEngines:
    def __init__(self, df):
        self.df = df
        self.ball_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']

    def engine_lstm_engineer(self):
        """Enfoque Deep Learning"""
        lookback = 10
        data_series = []
        for _, row in self.df[self.ball_cols].iterrows():
            vector = np.zeros(49)
            for val in row.values:
                vector[int(val)-1] = 1
            data_series.append(vector)
        
        X = np.array([data_series[i:i+lookback] for i in range(len(data_series)-lookback)])
        
        # Modelo Principal (Bolas)
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=False), input_shape=(lookback, 49)),
            Dense(49, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        train_limit = min(len(X), 500) # Entrenar con los últimos 500 para velocidad
        model.fit(X[-train_limit:], np.array(data_series[-train_limit:]), epochs=15, verbose=0, batch_size=16)
        
        last_seq = np.array(data_series[-lookback:]).reshape(1, lookback, 49)
        pred = model.predict(last_seq, verbose=0)[0]
        top_indices = pred.argsort()[-6:][::-1]
        
        # Predicción Reintegro (Heurística basada en tendencia reciente para velocidad)
        # Usamos la moda de los últimos 20 sorteos como proxy de "tendencia"
        reintegros_recent = self.df['r'].tail(50).value_counts().index.tolist()
        pred_r = int(reintegros_recent[0]) if reintegros_recent else 0
        
        # Devolver enteros nativos de Python
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
        
        # Reintegro: El que más sale (Frecuencia pura global)
        r_counts = self.df['r'].value_counts()
        most_common_r = int(r_counts.idxmax()) if not r_counts.empty else 0
        
        return top_balls, most_common_r

    def engine_game_theory(self):
        """Enfoque Anti-Humano"""
        while True:
            # Generación aleatoria ponderada para evitar patrones visuales
            nums = sorted(random.sample(range(1, 50), 6))
            
            # 1. Suma total entre 120 y 180 (distribución normal típica)
            if not (120 <= sum(nums) <= 180):
                continue
                
            # 2. No más de 2 números consecutivos
            consecutives = sum(1 for i in range(len(nums)-1) if nums[i+1] == nums[i]+1)
            if consecutives > 2:
                continue
                
            # 3. Evitar "cumpleaños" (demasiados números <= 31)
            low_nums = sum(1 for n in nums if n <= 31)
            if low_nums > 4: # La gente juega muchas fechas, jugar >31 es +EV
                continue
                
            # Reintegro Aleatorio (Máxima Entropía)
            r_val = random.randint(0, 9)
            return nums, r_val