import pandas as pd
import sys
from pathlib import Path



ruta_raiz = Path(__file__).parent.parent
sys.path.append(str(ruta_raiz))


from dominio.municipio import Municipio
from dominio.clasificacion_delito import (
    DelitoViolento, DelitoPatrimonial, DelitoFamiliar, DelitoSexual, DelitoAbstracto
)
from dominio.conteo_mensual import ConteoMensual, Mes
from dominio.registro_incidencia import RegistroIncidencia

# Clase extra para manejar delitos que no caen en tus 4 principales
class DelitoOtros(DelitoAbstracto):
    def calcular_peso_estadistico(self) -> int:
        return 1

def transformar_dataframe_a_objetos(df_filtrado: pd.DataFrame) -> list[RegistroIncidencia]:
    """Convierte filas de Pandas en un grafo de objetos de Dominio (POO)."""
    
    lista_objetos = []
    
    # iterrows() nos permite leer el dataframe fila por fila
    for index, row in df_filtrado.iterrows():
        
        # 1. Instanciar Entidad Base (Municipio)
        mun = Municipio(row['Cve_Municipio'], row['Municipio'])
        
        # 2. PATRÓN FACTORY: Decidir qué clase hija de Delito instanciar
        bien_juridico = row['Bien_juridico_afectado']
        tipo = row['Tipo_de_delito']
        subtipo = row['Subtipo_de_delito']
        modalidad = row['Modalidad']
        
        if bien_juridico in ['La vida y la Integridad corporal', 'Libertad personal']:
            delito = DelitoViolento(bien_juridico, tipo, subtipo, modalidad)
        elif bien_juridico == 'El patrimonio':
            delito = DelitoPatrimonial(bien_juridico, tipo, subtipo, modalidad)
        elif bien_juridico == 'La familia':
            delito = DelitoFamiliar(bien_juridico, tipo, subtipo, modalidad)
        elif bien_juridico == 'La libertad y la seguridad sexual':
            delito = DelitoSexual(bien_juridico, tipo, subtipo, modalidad)
        else:
            delito = DelitoOtros(bien_juridico, tipo, subtipo, modalidad)
            
        # 3. Ensamblar la clase principal
        registro = RegistroIncidencia(
            id_registro=index, # Usamos el índice de Pandas como ID temporal
            anio=2015,
            totalAnual=row['Total_Anual'],
            tasaAnual100k=row['Tasa_Anual_100k'],
            municipio=mun,
            clasificacion=delito
        )
        
        # 4. Inyectar la temporalidad (Los 12 meses)
        meses_str = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                     'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
        
        for i, nombre_mes in enumerate(meses_str, start=1):
            # i va del 1 al 12, lo que encaja perfecto con tu Enum Mes(1), Mes(2)...
            conteo = ConteoMensual(Mes(i), row[nombre_mes])
            registro.agregar_conteo(conteo)
            
        lista_objetos.append(registro)
        
    return lista_objetos