# 1. Usamos una imagen ligera de Python
FROM python:3.9-slim

# 2. Definimos la carpeta de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiamos el archivo de requisitos e instalamos (caché optimizada)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiamos todo el código de tu proyecto al contenedor
COPY . .

# 5. Exponemos el puerto de Streamlit (por defecto es 8501)
EXPOSE 8501

# 6. Comando para arrancar tu app
CMD ["streamlit", "run", "tu_archivo_principal.py", "--server.port=8501", "--server.address=0.0.0.0"]