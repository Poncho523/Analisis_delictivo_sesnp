from abc import ABC, abstractmethod

class DelitoAbstracto(ABC):
    """Clase base abstracta para la taxonomía criminal."""
    
    def __init__(self, bienJuridicoAfectado: str, tipoDelito: str, subtipoDelito: str, modalidad: str):
        self.bienJuridicoAfectado: str = bienJuridicoAfectado
        self.tipoDelito: str = tipoDelito
        self.subtipoDelito: str = subtipoDelito
        self.modalidad: str = modalidad

    @abstractmethod
    def calcular_peso_estadistico(self) -> int:
        """Método que cada subclase debe implementar obligatoriamente."""
        pass


class DelitoViolento(DelitoAbstracto):
    """Agrupa Homicidio, Secuestro y Extorsión."""
    
    def calcular_peso_estadistico(self) -> int:
        return 10  # Máxima prioridad estadística


class DelitoPatrimonial(DelitoAbstracto):
    """Agrupa Robo, Fraude, Daño a la propiedad."""
    
    def calcular_peso_estadistico(self) -> int:
        return 3   # Prioridad media


class DelitoFamiliar(DelitoAbstracto):
    """Agrupa Violencia Familiar y similares."""
    
    def calcular_peso_estadistico(self) -> int:
        return 6   # Prioridad alta por vulnerabilidad social


class DelitoSexual(DelitoAbstracto):
    """Agrupa Violación, Abuso, etc."""
    
    def calcular_peso_estadistico(self) -> int:
        return 9   # Prioridad crítica