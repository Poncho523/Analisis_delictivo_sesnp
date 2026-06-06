class Municipio:
    """Representa un municipio dentro de una entidad federativa.
    
    Atributos:
        claveMunicipio (int): Identificador único del municipio.
        nombre (str): Nombre oficial del municipio.
    """

    def __init__(self, claveMunicipio: int, nombre: str):
        self.claveMunicipio: int = claveMunicipio
        self.nombre: str = nombre