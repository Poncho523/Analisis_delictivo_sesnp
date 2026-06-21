FROM python:3.9-slim

WORKDIR /app

COPY . . 

RUN pip install --no-cache-dir -r librerias.txt

CMD ["streamlit", "run", "interfaz_visualizador/interfaz.py", "--server.port=8501"]