from dominio.municipio import Municipio
from dominio.clasificacion_delito import DelitoAbstracto
from dominio.conteo_mensual import ConteoMensual

class RegistroIncidencia:
    """Clase central que asocia geografía, taxonomía penal y temporalidad.
    
    Representa el evento criminal consolidado en un municipio específico 
    durante un año fiscal.
    
    Atributos:
        id_registro (int): Identificador único del registro de incidencia.
        anio (int): Año fiscal de registro del evento.
        totalAnual (int): Sumatoria de la frecuencia absoluta en el año.
        tasaAnual100k (float): Tasa de delitos ponderada por cada 100,000 habitantes.
        municipio (Municipio): Asociación estricta (1) al lugar de los hechos.
        clasificacion (ClasificacionDelito): Asociación estricta (1) al tipo de delito.
        conteos (list[ConteoMensual]): Composición (1..12) del desglose de casos por mes.
    """

    def __init__(self, id_registro: int, anio: int, totalAnual: int, tasaAnual100k: float, 
                 municipio: Municipio, clasificacion: DelitoAbstracto):
        self.id_registro: int = id_registro
        self.anio: int = anio
        self.totalAnual: int = totalAnual
        self.tasaAnual100k: float = tasaAnual100k
        
        self.municipio: Municipio = municipio
        self.clasificacion: DelitoAbstracto = clasificacion
        
        self.conteos: list[ConteoMensual] = []

    def agregar_conteo(self, conteo: ConteoMensual) -> None:
        """Agrega un conteo mensual validando las reglas de negocio del dominio.
        
        Raises:
            ValueError: Si se intenta exceder el límite lógico de 12 meses.
        """
        if len(self.conteos) >= 12:
            raise ValueError(f"El registro {self.id_registro} ya contiene 12 meses.")
        self.conteos.append(conteo)

    def obtener_mes_moda(self) -> ConteoMensual | None:
        """Identifica el mes con la mayor cantidad de casos dentro de este registro.
        
        Returns:
            ConteoMensual | None: El objeto mensual con más casos, o None si está vacío.
        """
        if not self.conteos:
            return None
        # Retorna el mes con el pico más alto de criminalidad para este registro
        return max(self.conteos, key=lambda c: c.cantidadCasos)

    def categorizar_nivel_riesgo(self, umbral_alto: float, umbral_medio: float) -> str:
        """Convierte la tasa numérica en una categoría para análisis de contingencia (Chi-cuadrada).
        
        Args:
            umbral_alto (float): Tasa mínima para considerar riesgo 'Alto'.
            umbral_medio (float): Tasa mínima para considerar riesgo 'Medio'.
            
        Returns:
            str: Categoría de riesgo ('Alto', 'Medio', 'Bajo').
        """
        if self.tasaAnual100k >= umbral_alto:
            return "Alto"
        elif self.tasaAnual100k >= umbral_medio:
            return "Medio"
        return "Bajo"
    def calcular_tendencia_semestral(self) -> str:
        """Compara el primer semestre vs el segundo para detectar alzas criminales."""
        if len(self.conteos) != 12:
            return "Datos Incompletos"
            
        casos_s1 = sum(c.cantidadCasos for c in self.conteos[:6])
        casos_s2 = sum(c.cantidadCasos for c in self.conteos[6:])
        
        if casos_s2 > casos_s1:
            return "Alza Criminal en Semestre 2"
        elif casos_s1 > casos_s2:
            return "Reducción Criminal en Semestre 2"
        else:
            return "Criminalidad Estable"