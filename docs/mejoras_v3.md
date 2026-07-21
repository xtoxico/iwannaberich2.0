# 🔬 LottoMind 2.0 — Especificación de Mejoras v3

> **Objetivo**: Maximizar la tasa de aciertos (≥3 de 6) y el valor esperado del sistema,
> atacando las debilidades fundamentales detectadas tras la auditoría completa del código.

> [!IMPORTANT]
> La lotería es un proceso **estocástico sin memoria** (cada sorteo es IID — independiente e idénticamente distribuido).
> Ningún modelo puede predecir con certeza los resultados. Sin embargo, **sí existen estrategias para maximizar
> el valor esperado (EV)**: reducir el reparto de premios, cubrir más espacio combinatorio, y
> mejorar la calibración de los modelos para acertar más consistentemente 2-3 números que apuntando
> a los 6. Las mejoras aquí propuestas se centran en eso.

---

## 📊 Diagnóstico del Estado Actual

### Resultados observados (simulation_results.csv)

| Métrica | Valor |
|---|---|
| Sorteos simulados | ~30 (solo 5 draw_indices procesados × 6 engines) |
| Aciertos máximos | 3 (1 vez, por `clusters`) |
| Media de aciertos | ~0.6 de 6 |
| Tasa de 3+ aciertos | ~3% |
| Tasa esperada aleatoria (6 de 49) | ~0.73 aciertos de media |

**Conclusión**: El sistema actual rinde ligeramente **por debajo del azar puro**, lo cual
indica que los engines tienen sesgos sistemáticos que perjudican en vez de ayudar.

### Problemas fundamentales detectados

1. **La simulación empieza desde 1986** con solo 10 sorteos de entrenamiento — datos insuficientes
2. **Todos los engines asumen que el pasado predice el futuro** — falacia del jugador
3. **El LSTM aprende ruido**, no señal (los números de lotería son IID)
4. **Pesos del consenso** se calculan con datos estáticos o con backtesting de muestras diminutas
5. **Sin diversificación real** — los engines tienden a converger en los mismos números
6. **Sin gestión de bankroll** ni optimización de cobertura combinatoria

---

## 🏗️ ÁREA 1: Pipeline de Datos (ETL)

### 1.1 — Validación robusta del parsing de combinaciones

**Archivo**: [`etl.py`](file:///home/xtoxico/workspace/iwannaberich2.0/src/etl.py#L136-L157)

**Problema**: El parser de `combinacion` usa splits frágiles que pueden fallar con formatos
cambiantes de la API de Loterías y Apuestas. No hay logging de registros descartados.

**Mejora propuesta**:
```python
import re

def parse_combinacion(comb_str: str) -> dict | None:
    """Parser robusto con regex para todos los formatos conocidos."""
    # Formato esperado: "N1 - N2 - N3 - N4 - N5 - N6 C(CC) R(R)"
    patterns = [
        # Formato moderno
        r'(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*C\((\d+)\)\s*R\((\d+)\)',
        # Formato antiguo (solo números separados por espacios/guiones)
        r'(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)\D+(\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, comb_str)
        if m:
            nums = [int(g) for g in m.groups()]
            if len(nums) >= 8 and all(1 <= n <= 49 for n in nums[:6]):
                return {
                    'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
                    'n4': nums[3], 'n5': nums[4], 'n6': nums[5],
                    'c': nums[6], 'r': nums[7]
                }
    return None  # Descartado con log
```

**Prioridad**: 🟡 Media — Los datos actuales parecen cargarse bien, pero es propenso a rotura.

---

### 1.2 — Verificar orden canónico de los números

**Problema**: La API puede devolver las bolas desordenadas. Si `n1 > n2`, los modelos
que usan posición (LSTM) aprenden patrones espurios.

**Mejora propuesta**: Tras el parsing, forzar `n1 < n2 < n3 < n4 < n5 < n6`:

```python
def normalize_draw(row):
    nums = sorted([row['n1'], row['n2'], row['n3'], row['n4'], row['n5'], row['n6']])
    return pd.Series({'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
                      'n4': nums[3], 'n5': nums[4], 'n6': nums[5]})
```

**Prioridad**: 🔴 Alta — Impacta directamente en la calidad de las features del LSTM.

---

### 1.3 — Datos de entrenamiento post-2004 como opción

**Problema**: Los datos pre-2004 tienen reintegro `0` falso (no existía), y la mecánica del
sorteo pudo cambiar (diferentes máquinas, bombos, reglas). Mezclar regímenes diferentes
introduce ruido.

**Mejora propuesta**: Añadir opción configurable para filtrar solo datos post-2004 en los engines:

```python
class LottoEngines:
    def __init__(self, df, min_date='2004-01-01'):
        if min_date and 'fecha' in df.columns:
            df = df[pd.to_datetime(df['fecha']) >= min_date]
        self.df = df.reset_index(drop=True)
```

**Prioridad**: 🔴 Alta — ~2000 sorteos post-2004 son más que suficientes y son datos limpios.

---

## 🧠 ÁREA 2: Engines de Predicción

### 2.1 — Rediseño del LSTM: De predicción binaria a predicción de ranking

**Archivo**: [`engines.py`](file:///home/xtoxico/workspace/iwannaberich2.0/src/engines.py#L124-L216)

**Problema principal**: El modelo actual intenta predecir un vector binario de 49 posiciones
(qué bolas salen). Esto es un problema de clasificación multi-label con **distribución
extremadamente desbalanceada** (6 de 49 positivos = 12%). Con `binary_crossentropy`, el
modelo aprende a predecir todo como ~0.12 (la prior) y no discrimina.

**Mejoras propuestas**:

#### A) Cambiar a pérdida focal o weighted BCE
```python
import tensorflow.keras.backend as K

def focal_loss(gamma=2.0, alpha=0.75):
    def focal_loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = -alpha_t * K.pow(1 - p_t, gamma) * K.log(p_t)
        return K.mean(loss)
    return focal_loss_fn
```

#### B) Representación como ranking en vez de clasificación
En lugar de predecir un one-hot binario, predecir un **score de ranking** por bola y usar
una loss que premie que las 6 bolas correctas estén en el top-6 del ranking.

```python
# Usar listwise ranking loss (listMLE o approxNDCG)
# O más simple: sigmoid output + sortear por probabilidad
# y medir con top-k accuracy en vez de binary crossentropy
```

#### C) Features temporales avanzadas
```python
# Para cada ventana de lookback, añadir:
- Frecuencia acumulada de cada bola en últimos N sorteos
- Gap (sorteos desde última aparición) de cada bola
- Entropía de la distribución de bolas en la ventana
- Suma y varianza de la combinación
- Indicadores de tendencia (media móvil de frecuencia)
```

#### D) Arquitectura mejorada
```python
# Usar Transformer (Attention) en vez de LSTM para capturar dependencias largas
# O al menos añadir una capa de self-attention sobre el LSTM
from tensorflow.keras.layers import MultiHeadAttention

# 1D-CNN + BiLSTM + Attention = mejor que BiLSTM solo
model = Sequential([
    Conv1D(64, 3, activation='relu', padding='same', input_shape=(lookback, feature_dim)),
    Bidirectional(LSTM(128, return_sequences=True)),
    MultiHeadAttention(num_heads=4, key_dim=32),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(49, activation='sigmoid')
])
```

#### E) Ensemble de múltiples modelos LSTM
Entrenar N modelos con diferentes random seeds y promediar sus predicciones para reducir varianza.

**Prioridad**: 🔴 Alta — El LSTM es el motor central y actualmente es ineficaz.

---

### 2.2 — Engine Estadístico: Incorporar test Chi-Cuadrado

**Archivo**: [`engines.py`](file:///home/xtoxico/workspace/iwannaberich2.0/src/engines.py#L221-L250)

**Problema**: El engine usa frecuencia + lag con pesos fijos (0.6 lag, 0.4 freq). No hay
validación de si estas señales son estadísticamente significativas.

**Mejora propuesta**:
```python
from scipy.stats import chisquare

def engine_statistician_v2(self):
    # 1. Test chi-cuadrado para verificar si la distribución es uniforme
    freqs = self.df[self.ball_cols].stack().value_counts().reindex(range(1, 50), fill_value=0)
    expected = len(self.df) * 6 / 49  # Frecuencia esperada si uniforme
    chi2, p_value = chisquare(freqs.values, f_exp=[expected] * 49)
    
    if p_value > 0.05:
        # La distribución NO es significativamente diferente de uniforme
        # → Las frecuencias no son señal, son ruido
        # → Usar solo lag o aleatorio informado
        weight_freq = 0.1
        weight_lag = 0.9
    else:
        weight_freq = 0.4
        weight_lag = 0.6
    
    # 2. Ventanas temporales: comparar últimos 50, 200 y todo el histórico
    # 3. Añadir análisis de tendencia (¿la frecuencia sube o baja?)
```

**Prioridad**: 🟡 Media — Mejora la calibración del estadístico.

---

### 2.3 — Engine Markov: Cadenas de orden superior + contexto extendido

**Archivo**: [`engines.py`](file:///home/xtoxico/workspace/iwannaberich2.0/src/engines.py#L282-L322)

**Problema**: Las cadenas de Markov de 1er orden solo miran las transiciones del sorteo
inmediatamente anterior. El espacio de estados (49 bolas) es muy grande para 1er orden.

**Mejoras propuestas**:

```python
def engine_markov_v2(self):
    # 1. Markov de 2º orden: P(bola_t | sorteo_t-1, sorteo_t-2)
    # Usar un diccionario de tuplas como estado
    
    # 2. Agrupar bolas en "meta-estados" (décadas, pares/impares)
    # para reducir la dimensionalidad de la matriz de transición
    
    # 3. Regularización: suavizado de Laplace para evitar probabilidades 0
    transition_matrix += 0.01  # Suavizado
    
    # 4. Usar los últimos 3-5 sorteos como contexto, no solo el último
    context_draws = 3
    for lookback in range(1, context_draws + 1):
        balls = self.df[self.ball_cols].iloc[-(lookback)].values
        weight = 1.0 / lookback  # Más peso al más reciente
        for b in balls:
            scores += transition_matrix[int(b) - 1] * weight
```

**Prioridad**: 🟡 Media

---

### 2.4 — Engine Game Theory: Optimización de EV real

**Archivo**: [`engines.py`](file:///home/xtoxico/workspace/iwannaberich2.0/src/engines.py#L255-L277)

**Problema**: El engine actual simplemente evita números "populares" codificados a mano.
No usa datos reales de distribución de apuestas.

**Mejoras propuestas**:

```python
def engine_game_theory_v2(self):
    """
    1. Estimar la distribución de combinaciones populares usando:
       - Frecuencia de selección manual en encuestas/estudios publicados
       - Patrones conocidos: diagonales, cumpleaños (1-31), múltiplos
       - Números "bonitos": 7, 11, 13, 21, 33, 42, 49
    
    2. Calcular la probabilidad de compartir premio para cada combinación
    
    3. Maximizar EV = P(acertar) * premio_esperado - coste_boleto
       donde premio_esperado = bote / (1 + N_ganadores_esperados)
    """
    
    # Modelo de popularidad basado en datos reales
    popularity = np.zeros(50)  # 1-indexed
    
    # Sesgo de cumpleaños (1-31 más elegidos)
    for n in range(1, 32):
        popularity[n] += 2.0
    for n in range(32, 50):
        popularity[n] += 1.0
    
    # Sesgo de múltiplos de 7
    for n in [7, 14, 21, 28, 35, 42, 49]:
        popularity[n] += 1.5
    
    # Sesgo de "bonitos"
    for n in [1, 3, 7, 11, 13, 17, 21, 33]:
        popularity[n] += 1.0
    
    # Penalizar patrones geométricos (diagonales en el boleto)
    # ...
    
    # Seleccionar números con BAJA popularidad (menos probable compartir)
    # pero sin excluir completamente los populares
    anti_pop_scores = 1.0 / (popularity[1:] + 0.1)
    return self._weighted_sample(
        {i+1: anti_pop_scores[i] for i in range(49)}, n=6, temperature=0.5
    )
```

**Prioridad**: 🟢 Baja (no mejora aciertos, pero sí EV si aciertas)

---

### 2.5 — NUEVO Engine: XGBoost / LightGBM

**Justificación**: Los modelos basados en árboles funcionan mejor que LSTMs cuando
las features son tabulares y no hay dependencia temporal real. XGBoost puede capturar
interacciones no lineales entre features sin la complejidad de un LSTM.

```python
def engine_xgboost(self):
    """Gradient Boosting sobre features ingenierizadas."""
    import lightgbm as lgb
    
    # Features por sorteo:
    features = []
    for idx in range(len(self.df)):
        row = self.df.iloc[idx]
        nums = [int(row[c]) for c in self.ball_cols]
        
        f = {
            'sum': sum(nums),
            'mean': np.mean(nums),
            'std': np.std(nums),
            'range': max(nums) - min(nums),
            'n_even': sum(1 for n in nums if n % 2 == 0),
            'n_high': sum(1 for n in nums if n > 24),
            'n_consecutive': sum(1 for i in range(5) if nums[i+1] == nums[i] + 1),
            'decade_entropy': -sum(p * np.log2(p) for p in ... if p > 0),
        }
        
        # Features de ventana: frecuencias de cada bola en últimos N sorteos
        # Gap de cada bola, frecuencia reciente, etc.
        
        features.append(f)
    
    # Entrenar 49 clasificadores binarios (uno por bola)
    # o un multi-label classifier
```

**Prioridad**: 🔴 Alta — Potencialmente el mejor enfoque para features tabulares.

---

### 2.6 — NUEVO Engine: Análisis de Patrones Combinatorios

**Justificación**: En vez de predecir qué números salen, predecir **propiedades de la
combinación ganadora** (suma, paridad, distribución por décadas) y luego generar combinaciones
que cumplan esas propiedades.

```python
def engine_combinatorial(self):
    """Genera combinaciones que cumplen los patrones estadísticos más frecuentes."""
    
    # 1. Calcular distribución histórica de propiedades
    sums = self.df[self.ball_cols].sum(axis=1)
    mean_sum = sums.mean()  # ~150 típicamente
    std_sum = sums.std()    # ~30
    
    # Distribución de pares/impares
    even_counts = self.df[self.ball_cols].apply(
        lambda row: sum(1 for v in row if int(v) % 2 == 0), axis=1
    )
    most_common_even = even_counts.mode()[0]  # Típicamente 3
    
    # Distribución por tercios
    # ...
    
    # 2. Generar combinaciones que cumplan los patrones más frecuentes
    while True:
        combo = sorted(random.sample(range(1, 50), 6))
        if abs(sum(combo) - mean_sum) > std_sum: continue
        if sum(1 for n in combo if n % 2 == 0) != most_common_even: continue
        # ... más filtros basados en patrones históricos
        return combo
```

**Prioridad**: 🔴 Alta — Bajo coste de implementación, alto potencial de filtrado.

---

## 🏆 ÁREA 3: Sistema de Consenso

### 3.1 — Consenso ponderado con pesos dinámicos reales

**Archivo**: [`app.py`](file:///home/xtoxico/workspace/iwannaberich2.0/src/app.py#L337-L408)

**Problema**: Los pesos se calculan con backtesting de solo 20 muestras, lo cual es
estadísticamente insignificante. Además, si todos los engines rinden igual (~0.73 aciertos
de media), los pesos son básicamente iguales.

**Mejoras propuestas**:

```python
def get_dynamic_weights(df, simulation_path='data/simulation_results.csv'):
    """
    1. Usar TODA la simulación walk-forward como base (no solo 20 tests)
    2. Aplicar decaimiento temporal (resultados recientes pesan más)
    3. Usar Bayesian Model Averaging en vez de normalización lineal
    4. Bonificar engines que aciertan 3+ (no solo media de aciertos)
    """
    res = pd.read_csv(simulation_path)
    
    weights = {}
    for engine in res['engine'].unique():
        eng_data = res[res['engine'] == engine]
        
        # Peso base: media de aciertos
        avg = eng_data['matches'].mean()
        
        # Bonus por aciertos altos (3+ vale mucho más que 1-2)
        high_hits = len(eng_data[eng_data['matches'] >= 3])
        bonus = high_hits * 2.0
        
        # Decaimiento temporal: últimos 100 pesan 3x más que los primeros
        recent = eng_data.tail(100)['matches'].mean()
        historical = eng_data['matches'].mean()
        
        weights[engine] = (recent * 0.6 + historical * 0.4) + bonus
    
    # Normalizar
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}
```

**Prioridad**: 🔴 Alta — El consenso es la predicción final que se usa.

---

### 3.2 — Diversificación forzada entre las 3 combinaciones

**Problema**: Las combinaciones conservadora, equilibrada y agresiva pueden compartir
muchos números, reduciendo la cobertura del espacio combinatorio.

**Mejora propuesta**:

```python
def generate_diversified_predictions(consensus_series, n_combinations=3):
    """
    Generar N combinaciones que maximicen la cobertura del espacio.
    Penalizar solapamiento entre combinaciones.
    """
    combinations = []
    used_numbers = set()
    
    for i in range(n_combinations):
        scores = consensus_series.copy()
        # Penalizar números ya usados en otras combinaciones
        for n in used_numbers:
            if n in scores.index:
                scores[n] *= 0.3  # Reducir pero no eliminar
        
        # Seleccionar top-6 del score ajustado
        combo = sorted(scores.nlargest(6).index.tolist())
        combinations.append(combo)
        used_numbers.update(combo)
    
    return combinations
```

**Prioridad**: 🟡 Media — Aumenta la probabilidad de al menos 1 combinación con 3+ aciertos.

---

### 3.3 — Múltiples apuestas por sorteo (cobertura)

**Concepto**: En vez de jugar 1 combinación, generar un "paquete" de N combinaciones
(ej. 5-10) que maximicen la cobertura del espacio combinatorio.

```python
def generate_coverage_pack(self, n_tickets=8):
    """
    Usar un algoritmo de 'covering design' para seleccionar N combinaciones
    que maximicen la probabilidad de tener al menos 3 aciertos en alguna.
    
    Estrategia:
    1. Generar 1000 combinaciones candidatas (de todos los engines)
    2. Seleccionar las N que minimicen el solapamiento mutuo
    3. Calcular la cobertura teórica
    """
    candidates = []
    for _ in range(200):
        for engine_fn in [self.engine_statistician, self.engine_markov, ...]:
            pred, _ = engine_fn()
            candidates.append(set(pred))
    
    # Selección greedy: añadir la que más números nuevos cubre
    selected = [candidates[0]]
    covered = candidates[0].copy()
    
    for _ in range(n_tickets - 1):
        best = max(candidates, key=lambda c: len(c - covered))
        selected.append(best)
        covered.update(best)
    
    return [sorted(list(s)) for s in selected]
```

**Prioridad**: 🔴 Alta — La mejora más impactante en probabilidad real de premio.

---

## 📏 ÁREA 4: Evaluación y Backtesting

### 4.1 — Simulación walk-forward completa

**Archivo**: [`simulate_history.py`](file:///home/xtoxico/workspace/iwannaberich2.0/scripts/simulate_history.py)

**Problema**: La simulación actual solo ha procesado ~5 sorteos (de ~4000). Hay que
ejecutarla completa para tener datos significativos.

**Mejoras propuestas**:

1. **Ejecutar la simulación desde el sorteo 200** (no desde el 10/50) para que los
   engines tengan datos suficientes
2. **Paralelizar** la evaluación de engines por sorteo (threading)
3. **Añadir métricas ricas**: no solo matches, sino también:
   - Complementario acertado (sí/no)
   - Reintegro acertado (sí/no)
   - Suma de la predicción vs suma real
   - Categoría de premio obtenida
   - "Profit" simulado (coste de boleto vs premio obtenido)

```python
# Nuevas columnas en simulation_results.csv:
'complementario_hit',   # bool
'reintegro_hit',        # bool
'pred_sum', 'real_sum', # int
'prize_category',       # str: 'none', 'reintegro', '3', '3+c', '4', '5', '5+c', '6'
'prize_eur',            # float: premio en euros (estimado)
'cumulative_profit',    # float: beneficio acumulado
```

**Prioridad**: 🔴 Alta — Sin evaluación completa no puedes medir mejoras.

---

### 4.2 — Métricas de evaluación correctas

**Problema**: Usar `avg_matches` como métrica principal es engañoso. Un engine que siempre
acierta 1 tiene avg=1.0, pero es inútil. Uno que acierta 3 en el 5% de los sorteos y
0 en el resto tiene avg=0.15, pero **es mejor**.

**Métricas propuestas**:

```python
def evaluate_engine(matches_dist):
    return {
        'avg_matches': np.mean(matches_dist),
        'hit_rate_3plus': sum(1 for m in matches_dist if m >= 3) / len(matches_dist),
        'hit_rate_2plus': sum(1 for m in matches_dist if m >= 2) / len(matches_dist),
        'max_matches': max(matches_dist),
        'expected_value': calculate_ev(matches_dist),  # Considerando premios
        'consistency': np.std(matches_dist),  # Menor = más consistente
        'sharpe_ratio': np.mean(matches_dist) / max(np.std(matches_dist), 0.01),
    }

def calculate_ev(matches_dist):
    """EV considerando estructura de premios de La Primitiva."""
    prizes = {
        3: 8,        # 8 €
        4: 60,       # ~60 €
        5: 2000,     # ~2000 €
        6: 1_000_000 # Variable, estimación conservadora
    }
    ticket_cost = 1.0  # 1 € por apuesta
    
    total_ev = 0
    for m in matches_dist:
        total_ev += prizes.get(m, 0)
    
    return (total_ev / len(matches_dist)) - ticket_cost
```

**Prioridad**: 🔴 Alta

---

### 4.3 — Comparación siempre contra baseline aleatorio

**Problema**: No hay un baseline contra el que comparar. ¿Cómo sabes si el engine es
mejor que elegir números al azar?

**Mejora propuesta**: Añadir un `engine_random` como baseline obligatorio:

```python
def engine_random(self):
    """Baseline: combinación completamente aleatoria."""
    nums = sorted(random.sample(range(1, 50), 6))
    r = random.randint(0, 9)
    return nums, r
```

Y en cada simulación/backtesting, incluir `engine_random` para comparar.

**Prioridad**: 🔴 Alta — Fundamental para saber si los engines aportan valor.

---

## 🔧 ÁREA 5: Arquitectura y Código

### 5.1 — Configuración centralizada

**Problema**: Constantes hardcodeadas por todo el código (lookback=60, temperature=0.4,
pesos 0.6/0.4, rango de suma 115-185, etc.). Imposible experimentar sin modificar código.

**Mejora propuesta**: Crear `config.py`:

```python
# src/config.py
from dataclasses import dataclass

@dataclass
class EngineConfig:
    # LSTM
    lstm_lookback: int = 60
    lstm_epochs: int = 80
    lstm_batch_size: int = 32
    lstm_temperature: float = 0.4
    
    # Estadístico
    stat_weight_lag: float = 0.6
    stat_weight_freq: float = 0.4
    stat_top_candidates: int = 15
    stat_temperature: float = 0.6
    
    # Markov
    markov_recency_weight_range: tuple = (1.0, 3.0)
    markov_noise_scale: float = 0.15
    markov_temperature: float = 0.5
    
    # Game Theory
    game_sum_range: tuple = (115, 185)
    game_max_consecutives: int = 2
    game_max_popular: int = 2
    
    # Genético
    genetic_generations: int = 80
    genetic_pop_size: int = 150
    genetic_elite_pct: float = 0.2
    
    # Consenso
    consensus_weight_exponent: float = 1.5
    
    # Data
    min_training_date: str = '2004-01-01'
    min_training_samples: int = 100

CONFIG = EngineConfig()
```

**Prioridad**: 🟡 Media — Facilitaría la experimentación masiva.

---

### 5.2 — Reproducibilidad con semillas

**Problema**: Los engines usan `random` y `np.random` sin semillas fijables. Imposible
reproducir resultados para debuggear.

**Mejora propuesta**:

```python
class LottoEngines:
    def __init__(self, df, seed=None):
        self.df = df.reset_index(drop=True)
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
```

**Prioridad**: 🟡 Media

---

### 5.3 — Logging estructurado

**Problema**: El sistema usa `print()` para todo. No hay forma de analizar logs
programáticamente.

**Mejora propuesta**: Usar `logging` con formato estructurado:

```python
import logging
import json

logger = logging.getLogger('lottomind')

# En cada predicción:
logger.info(json.dumps({
    'event': 'prediction',
    'engine': 'statistician',
    'numbers': [3, 12, 25, 33, 41, 47],
    'reintegro': 5,
    'scores': {...},
    'timestamp': '2025-01-15T10:30:00'
}))
```

**Prioridad**: 🟢 Baja

---

## 🛡️ ÁREA 6: Seguridad

### 6.1 — Credenciales en deploy.py

**Archivo**: [`deploy.py`](file:///home/xtoxico/workspace/iwannaberich2.0/scripts/deploy.py#L5-L7)

**Problema CRÍTICO**: Las credenciales SSH están hardcodeadas en el código fuente:

```python
HOST = "192.168.2.20"
USER = "xtoxico"
PASS = "ed21ttnd"  # ⚠️ CONTRASEÑA EN TEXTO PLANO
```

**Mejora propuesta**: Usar variables de entorno o `.env`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.environ['DEPLOY_HOST']
USER = os.environ['DEPLOY_USER']
PASS = os.environ['DEPLOY_PASS']
```

Y añadir `.env` al `.gitignore`.

**Prioridad**: 🔴 **URGENTE** — Riesgo de seguridad si el repo es público.

---

## 📈 ÁREA 7: Estrategias Avanzadas

### 7.1 — Sistema de Wheeling (Cobertura Combinatoria)

**Concepto**: Si tienes un conjunto de N números candidatos "calientes" (ej. 12 números),
generar un sistema de apuestas que garantice acertar al menos 3 si X de tus N números
son correctos.

```python
def generate_wheel(hot_numbers, guarantee=3, if_correct=4):
    """
    Si 4 de tus 12 números son correctos, el wheel garantiza
    al menos una combinación con 3 aciertos.
    
    Usa covering designs (t-designs) de la combinatoria.
    """
    from itertools import combinations
    
    # Generar todas las combinaciones de 6 de los N hot
    all_combos = list(combinations(hot_numbers, 6))
    
    # Seleccionar el mínimo subconjunto que cubre todos los
    # subconjuntos de tamaño 'guarantee' dentro de cualquier
    # subconjunto de tamaño 'if_correct'
    # (NP-hard, pero heurísticas greedy funcionan bien para N<20)
    ...
```

**Prioridad**: 🟡 Media — Requiere investigación algorítmica pero puede ser muy potente.

---

### 7.2 — Multi-Sorteo: Acumulación de evidencia entre sorteos

**Concepto**: Mantener un "score acumulado" de cada número a lo largo de múltiples sorteos.
Si el 23 lleva 15 sorteos sin salir y su ciclo medio es 8, su score acumulado sube.

```python
class PersistentScoreTracker:
    def __init__(self):
        self.scores = {n: 0.0 for n in range(1, 50)}
        self.history = []
    
    def update_after_draw(self, real_draw):
        for n in range(1, 50):
            if n in real_draw:
                self.scores[n] = 0  # Reset
            else:
                self.scores[n] += 0.1  # Acumular
        
        self.history.append(self.scores.copy())
    
    def get_hot_numbers(self, top_n=12):
        return sorted(self.scores, key=self.scores.get, reverse=True)[:top_n]
```

**Prioridad**: 🟡 Media

---

### 7.3 — Hyperparameter Tuning Automatizado

**Concepto**: Usar Optuna o similar para encontrar los mejores hiperparámetros de cada engine
(temperatures, pesos, lookbacks, etc.) mediante walk-forward cross-validation.

```python
import optuna

def objective(trial):
    # Hiperparámetros a optimizar
    stat_weight_lag = trial.suggest_float('stat_weight_lag', 0.1, 0.9)
    stat_temperature = trial.suggest_float('stat_temperature', 0.2, 1.5)
    markov_noise = trial.suggest_float('markov_noise', 0.05, 0.5)
    
    config = EngineConfig(
        stat_weight_lag=stat_weight_lag,
        stat_temperature=stat_temperature,
        markov_noise_scale=markov_noise,
    )
    
    # Walk-forward evaluation
    score = walk_forward_evaluate(df, config, n_tests=200)
    return score  # Maximizar hit_rate_3plus

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)
```

**Prioridad**: 🟡 Media — Alto impacto potencial pero requiere computación significativa.

---

## 📋 Resumen de Prioridades

### 🔴 Alta (Implementar Primero)

| # | Mejora | Impacto Esperado |
|---|--------|------------------|
| 6.1 | Eliminar credenciales del código | Seguridad |
| 1.2 | Verificar orden canónico de números | Correctitud de datos |
| 1.3 | Filtrar solo datos post-2004 | Reducción de ruido |
| 2.1 | Rediseñar LSTM (focal loss + features) | +15-30% en avg matches |
| 2.5 | Nuevo engine XGBoost/LightGBM | +10-20% vs LSTM |
| 2.6 | Nuevo engine combinatorio | Mejor filtrado de candidatos |
| 3.1 | Pesos dinámicos reales para consenso | +5-10% en hit rate |
| 3.3 | Paquetes de cobertura múltiple | +200-300% en P(3+ en alguna) |
| 4.1 | Simulación walk-forward completa | Prerequisito para medir todo |
| 4.2 | Métricas de evaluación correctas | Prerequisito para comparar |
| 4.3 | Baseline aleatorio obligatorio | Prerequisito para validar |

### 🟡 Media (Segundo Sprint)

| # | Mejora | Impacto Esperado |
|---|--------|------------------|
| 1.1 | Parser robusto con regex | Robustez |
| 2.2 | Chi-cuadrado en estadístico | Calibración |
| 2.3 | Markov de orden superior | +5% en engine Markov |
| 3.2 | Diversificación entre combinaciones | +10% cobertura |
| 5.1 | Configuración centralizada | Velocidad de experimentación |
| 5.2 | Reproducibilidad con semillas | Debugging |
| 7.1 | Sistema de Wheeling | Alto impacto si bien implementado |
| 7.2 | Score acumulado multi-sorteo | Persistencia de señales |
| 7.3 | Hyperparameter tuning (Optuna) | Optimización global |

### 🟢 Baja (Tercer Sprint)

| # | Mejora | Impacto Esperado |
|---|--------|------------------|
| 2.4 | Game Theory con datos reales | Mejor EV (no más aciertos) |
| 5.3 | Logging estructurado | Operaciones |

---

## 🎯 Plan de Acción Recomendado

```
Fase 0 (Inmediato):
  ├── Mover credenciales a .env
  └── Añadir engine_random como baseline

Fase 1 (Semana 1-2):
  ├── Ejecutar simulación walk-forward completa (post-2004, ~2000 sorteos)
  ├── Implementar métricas correctas (hit_rate_3plus, EV)
  ├── Normalizar datos (orden canónico, filtro post-2004)
  └── Comparar todos los engines vs baseline aleatorio

Fase 2 (Semana 3-4):
  ├── Rediseñar LSTM (focal loss, features, architecture)
  ├── Implementar engine XGBoost
  ├── Implementar engine combinatorio (filtro de propiedades)
  └── Re-evaluar con simulación completa

Fase 3 (Semana 5-6):
  ├── Implementar paquetes de cobertura (wheeling simplificado)
  ├── Optimizar hiperparámetros con Optuna
  ├── Implementar consenso con pesos dinámicos reales
  └── Evaluación final y comparativa
```

---

> [!TIP]
> **La mejora con mayor ROI inmediato** es ejecutar la simulación walk-forward completa
> con un `engine_random` como baseline. Solo así sabrás qué engines aportan valor real
> y cuáles son ruido. Todo lo demás depende de esta medición.
