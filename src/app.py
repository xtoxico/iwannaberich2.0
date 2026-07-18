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

.risk-conservative {
    border-left: 4px solid #22c55e;
    padding-left: 12px;
    margin: 8px 0;
}
.risk-balanced {
    border-left: 4px solid #f59e0b;
    padding-left: 12px;
    margin: 8px 0;
}
.risk-aggressive {
    border-left: 4px solid #ef4444;
    padding-left: 12px;
    margin: 8px 0;
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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "🧠 IA (LSTM)",
    "📊 Estadístico",
    "♟️ Estratega",
    "🔗 Markov",
    "🎯 Décadas",
    "🧬 Genético",
    "🗂️ Clústeres",
    "⏱️ Temporal",
    "🔮 Consenso",
    "📈 Rendimiento",
    "📉 Análisis"
])

# ── TAB 1: LSTM ───────────────────────────────────────────────────────────────
with tab1:
    st.header("Red Neuronal BiLSTM Profunda")
    st.write("Lookback de 30 sorteos · 2 capas BiLSTM · BatchNorm · Features enriquecidas · LR Scheduling.")
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
    st.write("Combina frecuencia histórica con retraso (lag) usando muestreo ponderado. Cada ejecución puede variar.")
    if st.button("📊 Calcular Probabilidades", key="btn_stat"):
        pred_stat, r_stat = engines.engine_statistician()
        draw_balls(pred_stat, reintegro=r_stat)

# ── TAB 3: ESTRATEGA ──────────────────────────────────────────────────────────
with tab3:
    st.header("Teoría de Juegos (EV Maximization)")
    st.write("Genera combinaciones anti-populares para maximizar el premio si aciertas.")
    if st.button("♟️ Generar Jugada Única", key="btn_game"):
        pred_game, r_game = engines.engine_game_theory()
        draw_balls(pred_game, reintegro=r_game)

# ── TAB 4: MARKOV ─────────────────────────────────────────────────────────────
with tab4:
    st.header("Cadenas de Markov")
    st.write("Transiciones ponderadas por recencia con ruido gaussiano para variabilidad.")
    if st.button("🔗 Calcular Transiciones", key="btn_markov"):
        pred_markov, r_markov = engines.engine_markov()
        draw_balls(pred_markov, reintegro=r_markov)

# ── TAB 5: DÉCADAS ────────────────────────────────────────────────────────────
with tab5:
    st.header("Análisis de Décadas")
    st.write("Selecciona de las décadas más frías, muestreando dentro de cada una.")
    if st.button("🎯 Analizar Décadas", key="btn_decades"):
        pred_dec, r_dec = engines.engine_decades()
        draw_balls(pred_dec, reintegro=r_dec)

# ── TAB 6: GENÉTICO ───────────────────────────────────────────────────────────
with tab6:
    st.header("Algoritmo Genético")
    st.write("80 generaciones · 150 individuos · Crossover 2 puntos · Mutación adaptativa.")
    if st.button("🧬 Evolucionar Población", key="btn_genetic"):
        pred_gen, r_gen = engines.engine_genetic()
        draw_balls(pred_gen, reintegro=r_gen)

# ── TAB 7: CLÚSTERES ──────────────────────────────────────────────────────────
with tab7:
    st.header("Análisis de Clústeres (K-Means)")
    st.write("Agrupa sorteos similares y pondera frecuencias por recencia dentro del clúster actual.")
    if st.button("🗂️ Identificar Clúster Actual", key="btn_clusters"):
        pred_clust, r_clust = engines.engine_clusters()
        draw_balls(pred_clust, reintegro=r_clust)

# ── TAB 8: TEMPORAL (NUEVO) ──────────────────────────────────────────────────
with tab8:
    st.header("⏱️ Patrones Temporales")
    st.write("Analiza ciclos de aparición, co-ocurrencias y patrones por día de sorteo (L/J/S).")
    if st.button("⏱️ Analizar Patrones", key="btn_temporal"):
        pred_temp, r_temp = engines.engine_temporal_patterns()
        draw_balls(pred_temp, reintegro=r_temp)

# ── TAB 9: CONSENSO ───────────────────────────────────────────────────────────
with tab9:
    st.header("🔮 El Oráculo — Consenso Ponderado (Adaptativo)")
    st.write("Combina **8 engines** usando pesos calculados dinámicamente según su rendimiento en Backtesting.")
    use_weights = st.checkbox("Usar Pesos Adaptativos (Backtesting)", value=True)

    if st.button("🏆 Generar Predicción Maestra", key="btn_consenso"):
        with st.spinner("Consultando oráculos..."):
            p_stat, r_stat = engines.engine_statistician()
            p_markov, r_markov = engines.engine_markov()
            p_dec, r_dec = engines.engine_decades()
            p_game, r_game = engines.engine_game_theory()
            p_gen, r_gen = engines.engine_genetic()
            p_clust, r_clust = engines.engine_clusters()
            p_temp, r_temp = engines.engine_temporal_patterns()
            
            engine_names = ['statistician', 'markov', 'decades', 'game_theory', 'genetic', 'clusters', 'temporal_patterns']
            weights = get_engine_weights(df, n_tests=20) if use_weights else {}
            if not weights:
                weights = {k: 1.0 for k in engine_names}

        # Mostrar predicciones individuales
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.caption(f"📊 Estadístico ({weights.get('statistician', 1):.2f})")
            draw_balls(p_stat, r_stat)
            st.caption(f"🧬 Genético ({weights.get('genetic', 1):.2f})")
            draw_balls(p_gen, r_gen)
            st.caption(f"⏱️ Temporal ({weights.get('temporal_patterns', 1):.2f})")
            draw_balls(p_temp, r_temp)
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

        # ── Consenso con ranking exponencial ──
        all_preds = [
            (p_stat, 'statistician'), (p_markov, 'markov'), (p_dec, 'decades'),
            (p_game, 'game_theory'), (p_gen, 'genetic'), (p_clust, 'clusters'),
            (p_temp, 'temporal_patterns')
        ]

        votes = {}
        for pred, name in all_preds:
            w = weights.get(name, 1)
            # Ranking exponencial: el peso del backtesting se eleva al cuadrado para separar más los buenos
            w_exp = w ** 1.5
            for ball in pred:
                votes[ball] = votes.get(ball, 0) + w_exp

        consenso_series = pd.Series(votes).sort_values(ascending=False)

        # ── Generar 3 combinaciones por nivel de riesgo ──
        st.subheader("🏆 Predicciones por Nivel de Riesgo")

        # CONSERVADORA: top-6 del consenso (máxima coincidencia entre engines)
        conservadora = sorted([int(x) for x in consenso_series.head(6).index.tolist()])

        # EQUILIBRADA: top-3 consenso + 3 muestreados del top-12
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

        # AGRESIVA: 2 del consenso + 4 de los menos votados pero presentes en al menos 1 engine
        top2 = [int(x) for x in consenso_series.head(2).index.tolist()]
        tail_candidates = {int(x): float(consenso_series[x]) for x in consenso_series.index[6:] if int(x) not in top2}
        if len(tail_candidates) >= 4:
            tail_keys = list(tail_candidates.keys())
            tail_vals = np.array(list(tail_candidates.values()))
            tail_probs = tail_vals / tail_vals.sum()
            extra_4 = list(np.random.choice(tail_keys, size=4, replace=False, p=tail_probs[:len(tail_keys)]))
        else:
            import random as rnd
            extra_4 = sorted(rnd.sample([n for n in range(1, 50) if n not in top2], 4))
        agresiva = sorted(top2 + [int(x) for x in extra_4])

        # Reintegro: distribución ponderada de todos los reintegros generados
        all_rs = [r_stat, r_markov, r_dec, r_game, r_gen, r_clust, r_temp]
        r_counts = pd.Series(all_rs).value_counts()
        r_probs = r_counts / r_counts.sum()
        consenso_r = int(np.random.choice(r_probs.index, p=r_probs.values))

        col_cons, col_eq, col_agr = st.columns(3)
        with col_cons:
            st.markdown("<div class='risk-conservative'>", unsafe_allow_html=True)
            st.markdown("**🟢 Conservadora** — Máximo consenso")
            draw_balls(conservadora, reintegro=consenso_r)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_eq:
            st.markdown("<div class='risk-balanced'>", unsafe_allow_html=True)
            st.markdown("**🟡 Equilibrada** — Top-3 + variación")
            draw_balls(equilibrada, reintegro=engines._predict_reintegro('consensus_balanced'))
            st.markdown("</div>", unsafe_allow_html=True)

        with col_agr:
            st.markdown("<div class='risk-aggressive'>", unsafe_allow_html=True)
            st.markdown("**🔴 Agresiva** — Apuesta diferencial")
            draw_balls(agresiva, reintegro=engines._predict_reintegro('consensus_aggressive'))
            st.markdown("</div>", unsafe_allow_html=True)

# ── TAB 10: RENDIMIENTO ────────────────────────────────────────────────────────
with tab10:
    st.header("📈 Backtesting — Rendimiento Histórico Real")
    n_tests_bt = st.slider("Número de sorteos a testear", 10, 100, 30, step=10)
    if st.button("🔬 Ejecutar Backtesting Completo"):
        from src.backtester import backtest_all_engines
        with st.spinner("Analizando..."):
            summary_df = backtest_all_engines(df, n_tests=n_tests_bt, skip_slow=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.success(f"🏆 Mejor motor: **{summary_df.iloc[0]['Engine']}**")

        # Gráfico de barras comparativo
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=summary_df['Engine'],
            y=summary_df['Avg. Aciertos (de 6)'],
            marker_color=['#22c55e' if i == 0 else '#3b82f6' for i in range(len(summary_df))],
            text=summary_df['Avg. Aciertos (de 6)'].round(3),
            textposition='outside'
        ))
        fig.update_layout(
            title="Comparativa de Rendimiento por Motor",
            xaxis_title="Motor", yaxis_title="Media de Aciertos (de 6)",
            template="plotly_dark", height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 11: ANÁLISIS (NUEVO) ──────────────────────────────────────────────────
with tab11:
    st.header("📉 Análisis del Histórico")

    # ── Frecuencia de números ──
    st.subheader("Frecuencia de aparición de cada número")
    ball_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    all_balls = df[ball_cols].stack()
    freq_counts = all_balls.value_counts().sort_index()
    freq_df = pd.DataFrame({'Número': freq_counts.index, 'Apariciones': freq_counts.values})

    fig_freq = px.bar(
        freq_df, x='Número', y='Apariciones',
        color='Apariciones', color_continuous_scale='blues',
        title="Frecuencia de cada número (1-49)"
    )
    fig_freq.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_freq, use_container_width=True)

    # ── Top 10 más y menos frecuentes ──
    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown("**🔥 Top 10 más frecuentes**")
        top10 = all_balls.value_counts().head(10)
        st.dataframe(
            pd.DataFrame({'Número': top10.index, 'Apariciones': top10.values}),
            hide_index=True, use_container_width=True
        )
    with col_bot:
        st.markdown("**❄️ Top 10 menos frecuentes**")
        bot10 = all_balls.value_counts().tail(10).sort_values()
        st.dataframe(
            pd.DataFrame({'Número': bot10.index, 'Apariciones': bot10.values}),
            hide_index=True, use_container_width=True
        )

    st.divider()

    # ── Distribución del Reintegro (solo post-2004) ──
    st.subheader("Distribución del Reintegro (solo datos fiables post-2004)")
    df_post2004 = df[pd.to_datetime(df['fecha']) >= '2004-01-01']
    r_counts = df_post2004['r'].value_counts().sort_index()
    fig_r = px.bar(
        x=r_counts.index, y=r_counts.values,
        labels={'x': 'Reintegro', 'y': 'Apariciones'},
        title="Frecuencia del Reintegro (post-2004)",
        color=r_counts.values, color_continuous_scale='reds'
    )
    fig_r.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(fig_r, use_container_width=True)

    st.divider()

    # ── Heatmap de co-ocurrencia ──
    st.subheader("Heatmap de Co-ocurrencia (Top 20 números)")
    top20_nums = all_balls.value_counts().head(20).index.tolist()
    cooc_matrix = pd.DataFrame(0, index=top20_nums, columns=top20_nums)
    for _, row in df[ball_cols].iterrows():
        draw = [int(v) for v in row.values if int(v) in top20_nums]
        for i in range(len(draw)):
            for j in range(i + 1, len(draw)):
                cooc_matrix.loc[draw[i], draw[j]] += 1
                cooc_matrix.loc[draw[j], draw[i]] += 1

    fig_heat = px.imshow(
        cooc_matrix, text_auto=True,
        color_continuous_scale='YlOrRd',
        title="Co-ocurrencia entre los 20 números más frecuentes"
    )
    fig_heat.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── Últimos 20 sorteos ──
    st.subheader("📋 Últimos 20 sorteos")
    last20 = df.tail(20).copy()
    last20['fecha'] = last20['fecha'].dt.strftime('%d/%m/%Y')
    display_cols = ['fecha'] + ball_cols + ['r', 'c']
    st.dataframe(last20[display_cols].iloc[::-1], hide_index=True, use_container_width=True)
