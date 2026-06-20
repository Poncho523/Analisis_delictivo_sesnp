from dominio.municipio import Municipio
from dominio.clasificacion_delito import DelitoAbstracto
from dominio.conteo_mensual import ConteoMensual

class RegistroIncidencia:
    """Clase central que asocia geografía, taxonomía penal y temporalidad."""

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
        """Agrega un conteo mensual validando las reglas de negocio del dominio."""
        if len(self.conteos) >= 12:
            raise ValueError(f"El registro {self.id_registro} ya contiene 12 meses.")
        self.conteos.append(conteo)

    def obtener_mes_moda(self) -> ConteoMensual | None:
        """Identifica el mes con la mayor cantidad de casos dentro de este registro."""
        if not self.conteos:
            return None
        return max(self.conteos, key=lambda c: c.cantidadCasos)

    def categorizar_nivel_riesgo(self, umbral_alto: float, umbral_medio: float) -> str:
        """Convierte la tasa numérica en una categoría para análisis de contingencia."""
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

    def calcular_indice_urgencia(self) -> float:
        """Cruza la gravedad del delito con su frecuencia demográfica."""
        return self.tasaAnual100k * self.clasificacion.calcular_peso_estadistico()

    def determinar_patron_ocurrencia(self) -> str:
        """Analiza la consistencia mensual del delito para planeación policial."""
        if not self.conteos:
            return "Desconocido"
            
        meses_activos = sum(1 for c in self.conteos if c.cantidadCasos > 0)
        
        if meses_activos >= 10:
            return "Crónico (Requiere vigilancia permanente)"
        elif meses_activos >= 4:
            return " Intermitente (Vigilancia regular)"
        elif meses_activos >= 1:
            return " Estacional / Esporádico (Operativos específicos)"
        else:
            return " Nulo"