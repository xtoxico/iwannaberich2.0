# app.py
import streamlit as st
import sys
import os

# Add project root to path so src imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.etl import actualizar_datos, cargar_datos, descargar_historico_completo, proximo_sorteo, nombre_dia_sorteo
from src.engines import LottoEngines, MODEL_PATH, TF_AVAILABLE
from src.backtester import get_engine_weights
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LottoMind 2.0", page_icon="🔮", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.ball {
    display: inline-block;
    width: 52px;
    height: 52px;
    line-height: 52px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #5ebcf9, #005c97);
    color: white;
    text-align: center;
    font-weight: 700;
    font-size: 16px;
    margin: 5px;
    box-shadow: 0 4px 12px rgba(0,92,151,0.4);
    transition: transform 0.15s ease;
}
.ball:hover { transform: scale(1.1); }

.ball-reintegro {
    display: inline-block;
    width: 52px;
    height: 52px;
    line-height: 52px;
    border-radius: 50%;
    background: radial-gradient(circle at 30% 30%, #ff6b6b, #b30000);
    color: white;
    text-align: center;
    font-weight: 700;
    font-size: 14px;
    margin: 5px 5px 5px 20px;
    box-shadow: 0 4px 12px rgba(179,0,0,0.4);
}

.draw-container {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
    padding: 12px 0;
}

.next-draw-banner {
    background: linear-gradient(135deg, #005c97, #363795);
    border-radius: 12px;
    padding: 14px 20px;
    color: white;
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def draw_balls(numbers, reintegro=None):
    html = "<div class='draw-container'>"
    for n in numbers:
        html += f"<div class='ball'>{n}</div>"
    if reintegro is not None:
        html += f"<div class='ball-reintegro'>R:{reintegro}</div>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_lstm(df_len: int, df_hash: str):
    """Caché del modelo LSTM para no reentrenar en cada interacción."""
    df = cargar_datos()
    engines = LottoEngines(df)
    return engines.engine_lstm_engineer()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1067/1067357.png", width=90)
    st.title("LottoMind 2.0")
    st.caption("Sistema Avanzado de Predicción Estocástica")

    st.markdown("---")
    if st.button("🔄 Actualizar Base de Datos", use_container_width=True):
        with st.spinner("Conectando..."):
            msg = actualizar_datos()
        st.success(msg)
        st.cache_resource.clear()
        time.sleep(1.5); st.rerun()

    if st.button("🌍 Descargar Histórico Completo", use_container_width=True):
        progress_bar = st.progress(0); status_text = st.empty()
        def update_progress(p, t): progress_bar.progress(p); status_text.text(t)
        with st.spinner("Descargando..."): msg = descargar_historico_completo(progress_callback=update_progress)
        st.success(msg); st.cache_resource.clear(); time.sleep(1.5); st.rerun()

    st.markdown("---")
    st.subheader("🧠 Estado de la IA")
    if os.path.exists(MODEL_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).strftime('%d/%m/%Y %H:%M')
        st.write(f"Último entrenamiento: **{mtime}**")
    else:
        st.warning("Modelo no entrenado localmente")
    
    if st.button("🚀 Re-entrenar IA Ahora", use_container_width=True):
        with st.spinner("Entrenando BiLSTM..."):
            df_local = cargar_datos()
            LottoEngines(df_local).engine_lstm_engineer(force_train=True)
        st.success("IA Actualizada"); time.sleep(1); st.rerun()

    st.markdown("---")
    st.caption("⚙️ La IA recuerda su modelo hasta que lo re-entrenes o actualices datos.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Cargar datos
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔮 Panel de Control LottoMind 2.0")

try:
    df = cargar_datos()
    if df.empty:
        st.warning("⚠️ **Histórico de datos vacío.** Pulsa 'Descargar Histórico' en la barra lateral.")
        st.stop()
    engines = LottoEngines(df)
except Exception as e:
    st.error(f"Error cargando datos: {e}"); st.stop()

# Banner próximo sorteo
proximo = proximo_sorteo()
if proximo:
    nombre_dia = nombre_dia_sorteo(proximo)
    st.markdown(
        f"<div class='next-draw-banner'>"
        f"🗓️ Próximo sorteo: <strong>{nombre_dia} {proximo.strftime('%d/%m/%Y')}</strong>"
        f"</div>",
        unsafe_allow_html=True
    )

# Métricas principales
col1, col2, col3, col4 = st.columns(4)
col1.metric("📊 Total Sorteos", f"{len(df):,}")
col2.metric("📅 Último Sorteo", df.iloc[-1]['fecha'].strftime('%d/%m/%Y'))
col3.metric("🔢 Números Posibles", "49")
col4.metric("🎯 Días de Sorteo", "L · J · S")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🧠 IA (LSTM)",
    "📊 Estadístico",
    "♟️ Estratega",
    "🔗 Markov",
    "🎯 Décadas",
    "🧬 Genético",
    "🗂️ Clústeres",
    "🔮 Consenso",
    "📈 Rendimiento"
])

# ── TAB 1: LSTM ───────────────────────────────────────────────────────────────
with tab1:
    st.header("Red Neuronal BiLSTM Profunda")
    st.write("Lookback de 30 sorteos · 2 capas BiLSTM · Early Stopping · Todos los datos históricos.")
    if not TF_AVAILABLE:
        st.warning("⚠️ **TensorFlow no disponible.** Se usará un motor de contingencia aleatorio.")
    
    col_run, col_info = st.columns([1, 2])
    with col_run:
        run_lstm = st.button("⚡ Ejecutar IA", key="btn_ai", use_container_width=True)

    if run_lstm:
        df_hash = str(len(df)) + str(df.iloc[-1]['fecha'])
        with st.spinner("Consultando/Entrenando BiLSTM..."):
            pred_ai, r_ai = get_cached_lstm(len(df), df_hash)
        st.success("✅ Predicción generada")
        draw_balls(pred_ai, reintegro=r_ai)

# ── TAB 2: ESTADÍSTICO ────────────────────────────────────────────────────────
with tab2:
    st.header("Análisis de Frecuencia y Retraso")
    if st.button("📊 Calcular Probabilidades", key="btn_stat"):
        pred_stat, r_stat = engines.engine_statistician()
        draw_balls(pred_stat, reintegro=r_stat)

# ── TAB 3: ESTRATEGA ──────────────────────────────────────────────────────────
with tab3:
    st.header("Teoría de Juegos (EV Maximization)")
    if st.button("♟️ Generar Jugada Única", key="btn_game"):
        pred_game, r_game = engines.engine_game_theory()
        draw_balls(pred_game, reintegro=r_game)

# ── TAB 4: MARKOV ─────────────────────────────────────────────────────────────
with tab4:
    st.header("Cadenas de Markov")
    if st.button("🔗 Calcular Transiciones", key="btn_markov"):
        pred_markov, r_markov = engines.engine_markov()
        draw_balls(pred_markov, reintegro=r_markov)

# ── TAB 5: DÉCADAS ────────────────────────────────────────────────────────────
with tab5:
    st.header("Análisis de Décadas")
    if st.button("🎯 Analizar Décadas", key="btn_decades"):
        pred_dec, r_dec = engines.engine_decades()
        draw_balls(pred_dec, reintegro=r_dec)

# ── TAB 6: GENÉTICO ───────────────────────────────────────────────────────────
with tab6:
    st.header("Algoritmo Genético")
    if st.button("🧬 Evolucionar Población", key="btn_genetic"):
        pred_gen, r_gen = engines.engine_genetic()
        draw_balls(pred_gen, reintegro=r_gen)

# ── TAB 7: CLÚSTERES ──────────────────────────────────────────────────────────
with tab7:
    st.header("Análisis de Clústeres (K-Means)")
    if st.button("🗂️ Identificar Clúster Actual", key="btn_clusters"):
        pred_clust, r_clust = engines.engine_clusters()
        draw_balls(pred_clust, reintegro=r_clust)

# ── TAB 8: CONSENSO ───────────────────────────────────────────────────────────
with tab8:
    st.header("🔮 El Oráculo — Consenso Ponderado (Adaptativo)")
    st.write("Combina todos los engines usando pesos calculados dinámicamente según su rendimiento en el Backtesting.")
    use_weights = st.checkbox("Usar Pesos Adaptativos (Backtesting)", value=True)

    if st.button("🏆 Generar Predicción Maestra", key="btn_consenso"):
        with st.spinner("Consultando oráculos..."):
            p_stat, r_stat = engines.engine_statistician()
            p_markov, r_markov = engines.engine_markov()
            p_dec, r_dec = engines.engine_decades()
            p_game, r_game = engines.engine_game_theory()
            p_gen, r_gen = engines.engine_genetic()
            p_clust, r_clust = engines.engine_clusters()
            
            weights = get_engine_weights(df, n_tests=20) if use_weights else {}
            if not weights:
                weights = {k: 1.0 for k in ['statistician', 'markov', 'decades', 'game_theory', 'genetic', 'clusters']}

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.caption(f"📊 Estadístico ({weights.get('statistician', 1):.2f})")
            draw_balls(p_stat, r_stat)
            st.caption(f"🧬 Genético ({weights.get('genetic', 1):.2f})")
            draw_balls(p_gen, r_gen)
        with col_b:
            st.caption(f"🔗 Markov ({weights.get('markov', 1):.2f})")
            draw_balls(p_markov, r_markov)
            st.caption(f"🗂️ Clústeres ({weights.get('clusters', 1):.2f})")
            draw_balls(p_clust, r_clust)
        with col_c:
            st.caption(f"🎯 Décadas ({weights.get('decades', 1):.2f})")
            draw_balls(p_dec, r_dec)
            st.caption(f"♟️ Estratega ({weights.get('game_theory', 1):.2f})")
            draw_balls(p_game, r_game)

        st.divider()

        votes = {}
        for ball in p_stat: votes[ball] = votes.get(ball, 0) + weights.get('statistician', 1)
        for ball in p_markov: votes[ball] = votes.get(ball, 0) + weights.get('markov', 1)
        for ball in p_dec: votes[ball] = votes.get(ball, 0) + weights.get('decades', 1)
        for ball in p_game: votes[ball] = votes.get(ball, 0) + weights.get('game_theory', 1)
        for ball in p_gen: votes[ball] = votes.get(ball, 0) + weights.get('genetic', 1)
        for ball in p_clust: votes[ball] = votes.get(ball, 0) + weights.get('clusters', 1)

        consenso_series = pd.Series(votes).sort_values(ascending=False).head(6)
        consenso = sorted([int(x) for x in consenso_series.index.tolist()])
        all_rs = [r_stat, r_markov, r_dec, r_game, r_gen, r_clust]
        consenso_r = max(set(all_rs), key=all_rs.count)

        st.subheader("🏆 Números Recomendados por Consenso Adaptativo:")
        draw_balls(consenso, reintegro=consenso_r)

# ── TAB 9: RENDIMIENTO ────────────────────────────────────────────────────────
with tab9:
    st.header("📈 Backtesting — Rendimiento Histórico Real")
    n_tests_bt = st.slider("Número de sorteos a testear", 10, 100, 30, step=10)
    if st.button("🔬 Ejecutar Backtesting Completo"):
        from src.backtester import backtest_all_engines
        with st.spinner("Analizando..."):
            summary_df = backtest_all_engines(df, n_tests=n_tests_bt, skip_slow=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.success(f"🏆 Mejor motor: **{summary_df.iloc[0]['Engine']}**")
