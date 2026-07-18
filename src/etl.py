# src/etl.py
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
import os
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'historico.csv')

# Días de sorteo de La Primitiva: Lunes=0, Jueves=3, Sábado=5
DIAS_SORTEO = {0, 3, 5}

def proximo_sorteo():
    """Devuelve la fecha del próximo sorteo de la Primitiva (Lunes, Jueves o Sábado)."""
    hoy = datetime.now()
    for i in range(7):
        candidato = hoy + timedelta(days=i)
        if candidato.weekday() in DIAS_SORTEO:
            return candidato
    return None  # No debería ocurrir nunca

def nombre_dia_sorteo(fecha: datetime) -> str:
    """Devuelve el nombre del día de sorteo en español."""
    nombres = {0: "Lunes", 3: "Jueves", 5: "Sábado"}
    return nombres.get(fecha.weekday(), "Desconocido")

def validar_datos(df):
    """Valida integridad de los datos: rango de números, reintegro, duplicados."""
    if df.empty:
        return df
    
    ball_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
    initial_len = len(df)
    
    # Validar rango de números (1-49)
    for col in ball_cols:
        df = df[(df[col] >= 1) & (df[col] <= 49)]
    
    # Validar reintegro (0-9)
    df = df[(df['r'] >= 0) & (df['r'] <= 9)]
    
    # Validar complementario (0-49, 0 = no disponible)
    if 'c' in df.columns:
        df = df[(df['c'] >= 0) & (df['c'] <= 49)]
    
    # Eliminar duplicados por fecha
    df = df.drop_duplicates(subset='fecha', keep='last')
    
    removed = initial_len - len(df)
    if removed > 0:
        logger.warning(f"Se eliminaron {removed} registros inválidos o duplicados.")
    
    return df


def cargar_datos():
    if not os.path.exists(DATA_PATH):
        # Crear estructura vacía si no existe
        df = pd.DataFrame(columns=['fecha', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'r', 'c'])
        df.to_csv(DATA_PATH, index=False)
    
    try:
        df = pd.read_csv(DATA_PATH)
        if df.empty:
            return pd.DataFrame(columns=['fecha', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'r', 'c'])
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = validar_datos(df)
        return df.sort_values('fecha').reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error cargando datos: {e}")
        return pd.DataFrame(columns=['fecha', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'r', 'c'])

def descargar_historico_completo(progress_callback=None):
    """Descarga todo el historial desde 1985 hasta la fecha actual"""
    year_actual = datetime.now().year
    todos_registros = []
    
    print("🔄 Iniciando descarga completa del histórico (1985-Presente)...")
    
    years = range(1985, year_actual + 1)
    total_years = len(years)
    
    for i, year in enumerate(years):
        if progress_callback:
            progress_callback(i / total_years, f"Descargando año {year}...")
            
        registros_year = descargar_anio(year)
        todos_registros.extend(registros_year)
        
    if todos_registros:
        df = pd.DataFrame(todos_registros)
        df = df.drop_duplicates(subset='fecha').sort_values('fecha')
        df.to_csv(DATA_PATH, index=False)
        return f"✅ Histórico completo descargado. {len(df)} sorteos registrados."
    else:
        return "⚠️ No se pudieron descargar datos. Verifica la conexión o el bloqueo del sitio oficial."

def descargar_anio(year):
    """Descarga los sorteos de un año específico con headers robustos"""
    url = f"https://www.loteriasyapuestas.es/servicios/buscadorSorteos?game_id=LAPR&celebrados=true&fechaInicioInclusiva={year}0101&fechaFinInclusiva={year}1231"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
        'Referer': 'https://www.loteriasyapuestas.es/es/la-primitiva',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Linux"',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'X-Requested-With': 'XMLHttpRequest'
    }
    
    registros = []
    session = requests.Session()
    
    try:
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            try:
                data = response.json()
            except Exception:
                return []
                
            if not data:
                return []
            
            for sorteo in data:
                try:
                    fecha_sorteo = datetime.strptime(sorteo['fecha_sorteo'], "%Y-%m-%d %H:%M:%S")
                    comb = sorteo['combinacion']
                    
                    parts = comb.replace('C(', '-').replace(') R(', '-').replace(')', '').split('-')
                    if len(parts) < 8: 
                         parts = [x.strip() for x in comb.replace('-', ' ').split() if x.strip().isdigit()]

                    nums = [int(p.strip()) for p in parts if p.strip().isdigit()]
                    
                    if len(nums) >= 6:
                        registro = {
                            'fecha': fecha_sorteo,
                            'n1': nums[0], 'n2': nums[1], 'n3': nums[2],
                            'n4': nums[3], 'n5': nums[4], 'n6': nums[5],
                            'c': nums[6] if len(nums) > 6 else 0,
                            'r': nums[7] if len(nums) > 7 else 0
                        }
                        registros.append(registro)
                except Exception:
                    pass
    except Exception:
        pass
        
    return registros

def actualizar_datos():
    """Descarga solo los sorteos nuevos desde la última fecha registrada con seguridad"""
    df = cargar_datos()
    
    if not df.empty:
        ultima_fecha = df.iloc[-1]['fecha']
        year_inicio = ultima_fecha.year
    else:
        return descargar_historico_completo()

    year_actual = datetime.now().year
    nuevos_registros = []

    for year in range(year_inicio, year_actual + 1):
        registros_anio = descargar_anio(year)
        for reg in registros_anio:
            if reg['fecha'] > ultima_fecha:
                nuevos_registros.append(reg)

    if nuevos_registros:
        df_new = pd.DataFrame(nuevos_registros)
        df_total = pd.concat([df, df_new]).drop_duplicates(subset='fecha').sort_values('fecha')
        df_total.to_csv(DATA_PATH, index=False)
        return f"✅ Se han añadido {len(nuevos_registros)} nuevos sorteos."
    else:
        return "✨ Los datos ya están actualizados o no se han encontrado nuevos sorteos."