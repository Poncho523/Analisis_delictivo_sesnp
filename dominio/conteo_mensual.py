from enum import Enum

class Mes(Enum):
    """Enumeración estricta de los meses del año para seguridad de tipos."""
    ENERO = 1
    FEBRERO = 2
    MARZO = 3
    ABRIL = 4
    MAYO = 5
    JUNIO = 6
    JULIO = 7
    AGOSTO = 8
    SEPTIEMBRE = 9
    OCTUBRE = 10
    NOVIEMBRE = 11
    DICIEMBRE = 12

class ConteoMensual:
    """Registra la frecuencia de un delito en un mes específico.
    
    Atributos:
        mes (Mes): El mes de ocurrencia, utilizando el tipo Enum seguro.
        cantidadCasos (int): Número absoluto de delitos ocurridos en ese mes.
    """

    def __init__(self, mes: Mes, cantidadCasos: int):
        self.mes: Mes = mes
        self.cantidadCasos: int = cantidadCasos

    def obtener_trimestre(self) -> int:
        """Calcula a qué trimestre del año pertenece este mes.
        
        Returns:
            int: Un valor del 1 al 4 representando el trimestre (Q1 a Q4).
        """
        return ((self.mes.value - 1) // 3) + 1