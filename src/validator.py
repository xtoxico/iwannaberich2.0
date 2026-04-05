import pandas as pd
import numpy as np
import os
from datetime import datetime

class Backtester:
    def __init__(self, df_historico, predictions_path):
        self.df_h = df_historico
        self.df_h['fecha'] = pd.to_datetime(self.df_h['fecha']).dt.strftime('%Y-%m-%d')
        self.preds_path = predictions_path

    def get_real_results(self, window_size=30):
        """Compara las predicciones guardadas con los resultados reales"""
        if not os.path.exists(self.preds_path):
            return pd.DataFrame()

        preds_df = pd.read_csv(self.preds_path)
        preds_df['fecha'] = pd.to_datetime(preds_df['fecha']).dt.strftime('%Y-%m-%d')
        
        # Merge con el histórico real
        merged = pd.merge(preds_df, self.df_h, on='fecha', suffixes=('_pred', '_real'))
        
        if merged.empty:
            return pd.DataFrame()

        results = []
        for _, row in merged.iterrows():
            nums_pred = {row['n1_pred'], row['n2_pred'], row['n3_pred'], row['n4_pred'], row['n5_pred'], row['n6_pred']}
            nums_real = {row['n1_real'], row['n2_real'], row['n3_real'], row['n4_real'], row['n5_real'], row['n6_real']}
            
            aciertos = len(nums_pred.intersection(nums_real))
            r_acierto = 1 if row['r_pred'] == row['r_real'] else 0
            
            results.append({
                'fecha': row['fecha'],
                'engine': row['engine'],
                'aciertos': aciertos,
                'reintegro_ok': r_acierto,
                'combinacion_real': list(nums_real)
            })
            
        return pd.DataFrame(results).tail(window_size * 3) # Window * 3 engines

    def calculate_metrics(self, results_df):
        if results_df.empty:
            return {}
            
        metrics = {}
        for engine in results_df['engine'].unique():
            engine_data = results_df[results_df['engine'] == engine]
            metrics[engine] = {
                'avg_hits': round(engine_data['aciertos'].mean(), 2),
                'max_hits': int(engine_data['aciertos'].max()),
                'r_success_rate': round(engine_data['reintegro_ok'].mean() * 100, 1),
                'total_draws': len(engine_data)
            }
        return metrics
