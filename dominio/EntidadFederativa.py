from dominio.municipio import Municipio

class EntidadFederativa:
    """Representa una entidad federativa (estado) geográfica.
    
    Atributos:
        claveEntidad (int): Identificador único de la entidad.
        nombre (str): Nombre oficial de la entidad federativa.
        municipios (list[Municipio]): Relación de composición (1..*). 
            Lista de municipios contenidos en esta entidad.
    """

    def __init__(self, claveEntidad: int, nombre: str):
        self.claveEntidad: int = claveEntidad
        self.nombre: str = nombre
        self.municipios: list[Municipio] = []