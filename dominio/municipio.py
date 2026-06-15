class Municipio:
    """Representa un municipio dentro de una entidad federativa."""

    def __init__(self, claveMunicipio: int, nombre: str):
        self.claveMunicipio: int = claveMunicipio
        self.nombre: str = nombre

    @staticmethod
    def clasificar_asentamiento(poblacion_total: int) -> str:
        """Regla de dominio (INEGI/CONAPO): Define la categoría demográfica.
        
        Args:
            poblacion_total (int): Número de habitantes del municipio.
            
        Returns:
            str: 'Rural', 'Urbano', o 'Megalópolis'.
        """
        if poblacion_total < 15000:
            return "Rural"
        elif poblacion_total <= 250000:
            return "Urbano"
        else:
            return "Megalópolis"