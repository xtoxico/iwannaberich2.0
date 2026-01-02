# 💰 iwannaberich2.0

Bienvenido a la versión 2.0 del sistema de predicción para **La Primitiva**. Este proyecto utiliza un enfoque híbrido combinando Deep Learning, Estadística Clásica y Teoría de Juegos para maximizar (teóricamente) las probabilidades de éxito o el valor esperado del premio.

## 🏗️ Estructura del Proyecto

- **src/engines.py**: Contiene los 3 cerebros (IA LSTM, Estadístico, Estratega).
- **src/etl.py**: Automatización de descarga de datos desde Loterías y Apuestas.
- **app.py**: Interfaz gráfica moderna construida con Streamlit.
- **data/**: Almacenamiento de históricos.

## 🚀 Instalación

   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Ejecución

Para iniciar la aplicación, ejecuta el siguiente comando desde la raíz del proyecto:

```bash
streamlit run src/app.py
```

O si estás dentro de la carpeta `src`:

```bash
streamlit run app.py
```

