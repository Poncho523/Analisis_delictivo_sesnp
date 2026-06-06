class ClasificacionDelito:
    """Define el tipo de delito penal de un incidente.
    
    Atributos:
        bienJuridicoAfectado (str): Categoría principal del bien protegido por la ley.
        tipoDelito (str): Clasificación general del crimen.
        subtipoDelito (str): Clasificación específica del crimen.
        modalidad (str): Forma exacta en la que se cometió el delito.
    """

    def __init__(self, bienJuridicoAfectado: str, tipoDelito: str, subtipoDelito: str, modalidad: str):
        self.bienJuridicoAfectado: str = bienJuridicoAfectado
        self.tipoDelito: str = tipoDelito
        self.subtipoDelito: str = subtipoDelito
        self.modalidad: str = modalidad