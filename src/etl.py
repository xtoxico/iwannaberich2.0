# src/etl.py
import pandas as pd
import requests
import json
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'historico.csv')

def cargar_datos():
    if not os.path.exists(DATA_PATH):
        # Crear estructura vacía si no existe
        df = pd.DataFrame(columns=['fecha', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'r', 'c'])
        df.to_csv(DATA_PATH, index=False)
    
    df = pd.read_csv(DATA_PATH)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df.sort_values('fecha')

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
        return "⚠️ No se pudieron descargar datos."

def descargar_anio(year):
    """Descarga los sorteos de un año específico"""
    url = f"https://www.loteriasyapuestas.es/servicios/buscadorSorteos?game_id=LAPR&celebrados=true&fechaInicioInclusiva={year}0101&fechaFinInclusiva={year}1231"
    
    # Headers que funcionaron en la prueba de bypass
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Referer': 'https://www.loteriasyapuestas.es/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'X-Requested-With': 'XMLHttpRequest'
    }
    
    registros = []
    session = requests.Session() # Usar sesion para mantener cookies/conexión
    
    try:
        response = session.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if not data:
                print(f"⚠️ Año {year}: JSON vacío o sin datos.")
            
            for sorteo in data:
                try:
                    fecha_sorteo = datetime.strptime(sorteo['fecha_sorteo'], "%Y-%m-%d %H:%M:%S")
                    comb = sorteo['combinacion']
                    
                    # Debug del formato si falla
                    # print(f"Procesando: {comb}") 

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
                except Exception as e:
                    # print(f"Error parseando sorteo {year}: {e}")
                    pass
        else:
            print(f"⚠️ Año {year}: Error HTTP {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error descargando año {year}: {e}")
        
    return registros

def actualizar_datos():
    """Descarga solo los sorteos nuevos desde la última fecha registrada"""
    df = cargar_datos()
    
    if not df.empty:
        ultima_fecha = df.iloc[-1]['fecha']
        year_inicio = ultima_fecha.year
    else:
        return descargar_historico_completo()

    year_actual = datetime.now().year
    nuevos_registros = []

    print(f"🔄 Buscando actualizaciones desde {year_inicio}...")

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
        return "✨ Los datos ya están actualizados."