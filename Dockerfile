FROM python:3.9-slim

WORKDIR /app

# Corregido: copiamos tu archivo real 'librerias.txt'
COPY librerias.txt .
RUN pip install --no-cache-dir -r librerias.txt

COPY . .

EXPOSE 8501

# Corregido: la I mayúscula en 'Interfaz_visualizador'
CMD ["streamlit", "run", "Interfaz_visualizador/interfaz.py", "--server.port=8501", "--server.address=0.0.0.0"]