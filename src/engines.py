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
        
        # Prepare X and y
        X = np.array([data_series[i:i+lookback] for i in range(len(data_series)-lookback)])
        y = np.array(data_series[lookback:])
        
        # Simulación rápida (Entrenar cada vez es lento, en prod deberías guardar el modelo .h5)
        # Para la demo, hacemos un entrenamiento "express"
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=False), input_shape=(lookback, 49)),
            Dense(49, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        # Entrenamos solo con los ultimos 200 sorteos para que la UI no se congele
        
        # Ensure we don't slice out of bounds if data is small
        train_limit = min(len(X), 200)
        if train_limit > 0:
            model.fit(X[-train_limit:], y[-train_limit:], epochs=10, verbose=0, batch_size=16)
        
        last_seq = np.array(data_series[-lookback:]).reshape(1, lookback, 49)
        pred = model.predict(last_seq, verbose=0)[0]
        top_indices = pred.argsort()[-6:][::-1]
        # Devolver enteros nativos de Python para evitar np.int64 en la UI
        return [int(i+1) for i in sorted(top_indices)]

    def engine_statistician(self):
        """Enfoque Frecuencia + Retraso"""
        freqs = self.df[self.ball_cols].stack().value_counts().sort_index()
        # Rellenar ceros si falta algún número
        freqs = freqs.reindex(range(1, 50), fill_value=0)
        
        last_seen = {}
        total = len(self.df)
        for n in range(1, 50):
            matches = self.df[self.df[self.ball_cols].isin([n]).any(axis=1)]
            last_seen[n] = total - matches.index[-1] if not matches.empty else total
            
        scores = {}
        max_freq = freqs.max()
        max_lag = max(last_seen.values())
        
        for n in range(1, 50):
            # Score = 70% peso al retraso, 30% a la frecuencia
            norm_freq = freqs[n] / max_freq
            norm_lag = last_seen[n] / max_lag
            scores[n] = (norm_lag * 0.7) + (norm_freq * 0.3)
            
        return sorted(sorted(scores, key=scores.get, reverse=True)[:6])

    def engine_game_theory(self):
        """Enfoque Anti-Humano"""
        while True:
            nums = sorted(random.sample(range(1, 50), 6))
            if not (120 <= sum(nums) <= 180): continue # Suma típica
            if len([n for n in nums if n > 31]) < 3: continue # Anti-cumpleaños
            if any(nums[i] == nums[i-1] + 1 for i in range(1, 6)): continue # No consecutivos
            return nums