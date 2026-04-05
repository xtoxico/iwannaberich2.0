# app.py
import streamlit as st
import sys
import os

# Add project root to path so src imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import plotly.express as px
from src.etl import actualizar_datos, cargar_datos, descargar_historico_completo
from src.engines import LottoEngines, PREDICTIONS_PATH, MODEL_PATH, TENSORFLOW_AVAILABLE
from src.validator import Backtester
import time
from datetime import datetime

# Configuración de página
st.set_page_config(page_title="LottoMind 2.0", page_icon="🔮", layout="wide")

# CSS personalizado
st.markdown("""
<style>
.ball {
    display: inline-block;
    width: 50px; height: 50px; line-height: 50px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #5ebcf9, #005c97);
    color: white; text-align: center; font-weight: bold; margin: 5px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
}
.metric-card {
    background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #005c97;
}
</style>
""", unsafe_allow_html=True)

def draw_balls(numbers, reintegro=None):
    html = "<div>"
    for n in numbers: html += f"<div class='ball'>{n}</div>"
    if reintegro is not None:
        html += f"<div class='ball' style='background: radial-gradient(circle at 30% 30%, #ff4b4b, #b30000); margin-left:15px;'>R: {reintegro}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1067/1067357.png", width=100)
    st.title("I wanna Be rich v2.0")
    
    st.subheader("⚙️ Configuración")
    backtest_window = st.slider("Ventana de Backtesting (Sorteos)", 5, 100, 30)
    
    if st.button("🔄 Actualizar Datos"):
        with st.spinner("Conectando..."): msg = actualizar_datos()
        st.success(msg)
        time.sleep(1); st.rerun()

    if st.button("🌍 Descarga Completa"):
        progress_bar = st.progress(0); status_text = st.empty()
        def up(p, t): progress_bar.progress(p); status_text.text(t)
        with st.spinner("Descargando..."): msg = descargar_historico_completo(progress_callback=up)
        st.success(msg); time.sleep(1); st.rerun()

    st.markdown("---")
    st.subheader("🧠 Estado de la IA")
    if os.path.exists(MODEL_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%d/%m/%Y %H:%M')
        st.write(f"Último entrenamiento: **{mtime}**")
    else:
        st.warning("Modelo no entrenado")
    
    if st.button("🚀 Re-entrenar IA Ahora"):
        with st.spinner("Entrenando..."):
            df = cargar_datos()
            LottoEngines(df).engine_lstm_engineer(force_train=True)
        st.success("IA Actualizada"); time.sleep(1); st.rerun()

# --- MAIN ---
st.title("Panel de Control de Predicción")

df = cargar_datos()

if df.empty:
    st.warning("⚠️ **Histórico de datos vacío.** No se puede realizar el análisis sin datos.")
    st.info("Por favor, utiliza el botón **'🌍 Descarga Completa'** en la barra lateral para bajar el historial desde 1985.")
    st.stop()

try:
    engines = LottoEngines(df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sorteos Analizados", len(df))
    if not df.empty:
        col2.metric("Última Fecha", df.iloc[-1]['fecha'].strftime('%d-%m-%Y'))
    col3.metric("Próximo Sorteo", engines.get_next_draw_date())
    
except Exception as e:
    st.error(f"Error analizando datos: {e}"); st.stop()

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🧠 IA (LSTM)", "📊 Estadístico", "♟️ Estratega", "🔮 Consenso", "🔬 Backtesting"])

with tab1:
    st.header("Red Neuronal LSTM")
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ **TensorFlow no disponible.** Esta función requiere Python 3.12 o inferior (actualmente 3.14). Se usará un motor de contingencia.")
    if st.button("Ver Predicción IA"):
        with st.spinner("Consultando oráculo..."):
            pred, r = engines.get_locked_prediction('IA')
        draw_balls(pred, r)
        st.caption("Esta predicción está 'bloqueada' para el próximo sorteo.")

with tab2:
    st.header("Análisis de Frecuencia")
    if st.button("Ver Predicción Estadística"):
        pred, r = engines.get_locked_prediction('Estadistico')
        draw_balls(pred, r)

with tab3:
    st.header("Teoría de Juegos")
    if st.button("Ver Predicción Estratega"):
        pred, r = engines.get_locked_prediction('Estratega')
        draw_balls(pred, r)

with tab4:
    st.header("El Oráculo (Consenso)")
    if st.button("Generar Predicción Maestra"):
        p1, r1 = engines.get_locked_prediction('IA')
        p2, r2 = engines.get_locked_prediction('Estadistico')
        p3, r3 = engines.get_locked_prediction('Estratega')
        
        all_nums = p1 + p2 + p3
        c_series = pd.Series(all_nums).value_counts().head(6)
        consenso = sorted([int(x) for x in c_series.index.tolist()])
        all_rs = [r1, r2, r3]
        consenso_r = max(set(all_rs), key=all_rs.count)
        
        st.subheader("🏆 Números Recomendados:")
        draw_balls(consenso, reintegro=consenso_r)

with tab5:
    st.header("Validación Histórica Real")
    validator = Backtester(df, PREDICTIONS_PATH)
    results = validator.get_real_results(window_size=backtest_window)
    
    if results.empty:
        st.info("No hay predicciones guardadas que coincidan con sorteos reales pasados todavía.")
        st.write("Las predicciones se guardan automáticamente cuando las consultas en las pestañas anteriores.")
    else:
        metrics = validator.calculate_metrics(results)
        
        # Mostrar métricas por engine
        cols = st.columns(len(metrics))
        for i, (engine, data) in enumerate(metrics.items()):
            with cols[i]:
                st.subheader(f"Engine: {engine}")
                st.metric("Aciertos Promedio", data['avg_hits'])
                st.metric("% Reintegro", f"{data['r_success_rate']}%")
        
        st.divider()
        st.subheader("Detalle de los últimos sorteos")
        st.dataframe(results[['fecha', 'engine', 'aciertos', 'reintegro_ok']], use_container_width=True)
        
        # Gráfico de evolución
        fig = px.line(results, x='fecha', y='aciertos', color='engine', title="Evolución de Aciertos")
        st.plotly_chart(fig, use_container_width=True)