FROM python:3.12
WORKDIR /app
COPY librerias.txt .
RUN pip install -r librerias.txt
COPY . .
CMD ["python", "main.py"]