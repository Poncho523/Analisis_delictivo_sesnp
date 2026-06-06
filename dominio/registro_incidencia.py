from municipio import Municipio
from clasificacion_delito import ClasificacionDelito
from conteo_mensual import ConteoMensual

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
                 municipio: Municipio, clasificacion: ClasificacionDelito):
        self.id_registro: int = id_registro
        self.anio: int = anio
        self.totalAnual: int = totalAnual
        self.tasaAnual100k: float = tasaAnual100k
        
        self.municipio: Municipio = municipio
        self.clasificacion: ClasificacionDelito = clasificacion
        
        self.conteos: list[ConteoMensual] = []