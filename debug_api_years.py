import requests
import json
from datetime import datetime

# Mimic the browser headers we added
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.loteriasyapuestas.es/',
    'Accept-Language': 'es-ES,es;q=0.9',
    'Connection': 'keep-alive'
}

years = [2025, 2026]

for year in years:
    print(f"\n--- Testing Year {year} ---")
    url = f"https://www.loteriasyapuestas.es/servicios/buscadorSorteos?game_id=LAPR&celebrados=true&fechaInicioInclusiva={year}0101&fechaFinInclusiva={year}1231"
    
    try:
        print(f"Requesting: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"Records found: {len(data)}")
                if len(data) > 0:
                    print("First record sample:", json.dumps(data[0], indent=2))
            except json.JSONDecodeError:
                print("Failed to decode JSON. Response text preview:")
                print(response.text[:200])
        else:
            print("Response text preview:")
            print(response.text[:200])
            
    except Exception as e:
        print(f"Error: {e}")
