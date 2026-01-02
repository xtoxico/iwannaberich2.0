# app.py
import streamlit as st
import sys
import os

# Add project root to path so src imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import plotly.express as px
from src.etl import actualizar_datos, cargar_datos, descargar_historico_completo
from src.engines import LottoEngines
import time

# Configuración de página
st.set_page_config(page_title="LottoMind 2.0", page_icon="🔮", layout="wide")

# CSS personalizado para que las bolas se vean bonitas
st.markdown("""
<style>
.ball {
    display: inline-block;
    width: 50px;
    height: 50px;
    line-height: 50px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #5ebcf9, #005c97);
    color: white;
    text-align: center;
    font-weight: bold;
    margin: 5px;
    box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
}
.metric-card {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #005c97;
}
</style>
""", unsafe_allow_html=True)

def draw_balls(numbers, reintegro=None):
    html = "<div>"
    for n in numbers:
        html += f"<div class='ball'>{n}</div>"
    
    if reintegro is not None:
        html += f"<div class='ball' style='background: radial-gradient(circle at 30% 30%, #ff4b4b, #b30000); margin-left: 15px;'>R: {reintegro}</div>"
        
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1067/1067357.png", width=100)
    st.title("LottoMind v2.0")
    st.write("Sistema Avanzado de Predicción Estocástica")
    
    if st.button("🔄 Actualizar Base de Datos"):
        with st.spinner("Conectando con Loterías y Apuestas..."):
            msg = actualizar_datos()
        st.success(msg)
        time.sleep(2)
        st.rerun()

    st.markdown("---")
    if st.button("🌍 Descargar Histórico (1985-Now)"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress, text):
            progress_bar.progress(progress)
            status_text.text(text)
            
        with st.spinner("Descargando 40 años de historia..."):
            msg = descargar_historico_completo(progress_callback=update_progress)
        
        progress_bar.empty()
        status_text.empty()
        st.success(msg)
        time.sleep(2)
        st.rerun()

# --- MAIN ---
st.title("Panel de Control de Predicción")

# Cargar datos
try:
    df = cargar_datos()
    engines = LottoEngines(df)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sorteos Analizados", len(df))
    col2.metric("Última Fecha", df.iloc[-1]['fecha'].strftime('%d-%m-%Y'))
    col3.metric("Entropía del Sistema", "Alta") # Decorativo
    
except Exception as e:
    st.error(f"Error cargando datos: {e}. Por favor pulsa Actualizar en el menú.")
    st.stop()

st.divider()

# TABS para los enfoques
tab1, tab2, tab3, tab4 = st.tabs(["🧠 El Ingeniero (IA)", "📊 El Estadístico", "♟️ El Estratega", "🔮 Consenso"])

with tab1:
    st.header("Red Neuronal LSTM")
    st.write("Busca patrones no lineales en secuencias temporales.")
    if st.button("Ejecutar IA", key="btn_ai"):
        with st.spinner("Entrenando red neuronal..."):
            pred_ai, r_ai = engines.engine_lstm_engineer()
        st.success("Predicción Generada")
        draw_balls(pred_ai, reintegro=r_ai)
        st.caption("Esta predicción se basa en las últimas 10 secuencias.")

with tab2:
    st.header("Análisis de Frecuencia y Retraso")
    st.write("Maximiza la probabilidad basada en la Ley de los Grandes Números.")
    if st.button("Calcular Probabilidades", key="btn_stat"):
        pred_stat, r_stat = engines.engine_statistician()
        draw_balls(pred_stat, reintegro=r_stat)
        
        # Gráfico bonito
        st.subheader("Mapa de Calor de Frecuencia")
        freqs = df[['n1','n2','n3','n4','n5','n6']].stack().value_counts().reset_index()
        freqs.columns = ['Bola', 'Apariciones']
        fig = px.bar(freqs.head(10), x='Bola', y='Apariciones', color='Apariciones', title="Top 10 Bolas Calientes")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Teoría de Juegos (EV Maximization)")
    st.write("Combinaciones diseñadas para ser únicas y evitar compartir el premio.")
    if st.button("Generar Jugada Única", key="btn_game"):
        pred_game, r_game = engines.engine_game_theory()
        draw_balls(pred_game, reintegro=r_game)
        st.info("Esta combinación cumple: Suma equilibrada, anti-cumpleaños y dispersión de decenas.")

with tab4:
    st.header("El Oráculo (Consenso)")
    if st.button("Generar Predicción Maestra"):
        p1, r1 = engines.engine_lstm_engineer()
        p2, r2 = engines.engine_statistician()
        p3, r3 = engines.engine_game_theory()
        
        st.write("🤖 **IA:**", f"{str(p1)} R:{r1}")
        st.write("📊 **Estadística:**", f"{str(p2)} R:{r2}")
        st.write("♟️ **Estrategia:**", f"{str(p3)} R:{r3}")
        
        all_nums = p1 + p2 + p3
        # Ensure we work with native ints for the Consensus
        consenso_series = pd.Series(all_nums).value_counts().head(6)
        consenso = sorted([int(x) for x in consenso_series.index.tolist()])
        
        # Consenso Reintegro (Moda)
        all_rs = [r1, r2, r3]
        consenso_r = max(set(all_rs), key=all_rs.count)
        
        st.subheader("🏆 Números Recomendados:")
        draw_balls(consenso, reintegro=consenso_r)