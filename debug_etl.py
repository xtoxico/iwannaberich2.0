import requests
import json

url = "https://www.loteriasyapuestas.es/servicios/buscadorSorteos?game_id=LAPR&celebrados=true&fechaInicioInclusiva=20240101&fechaFinInclusiva=20241231"

print(f"Testing URL: {url}")
try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    print(f"Status Code: {response.status_code}")
    print("Preview of content:")
    print(response.text[:500])
    
    data = response.json()
    print("\nJSON Decode Successful.")
    print(f"Items found: {len(data)}")
except Exception as e:
    print(f"\nError: {e}")
