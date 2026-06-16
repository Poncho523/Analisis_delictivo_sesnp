--Creamos la vista materializada
 CREATE MATERIALIZED VIEW vw_delitos_nivelados AS
WITH poblacion_agrupada AS (
    -- Primero agrupamos las 4 filas por municipio para tener un único total
    SELECT
        "Cve_Municipio",
        SUM("POB_TOTAL") AS poblacion_total
    FROM poblacion_cruda
    GROUP BY "Cve_Municipio"
)
SELECT
    d.*,
    COALESCE(p.poblacion_total, 0) AS "POB_TOTAL",
    -- Calculamos la tasa por cada 100k habitantes
    CASE
        WHEN COALESCE(p.poblacion_total, 0) > 0 THEN
            ROUND((d."Total_Anual"::numeric / p.poblacion_total) * 100000, 2)
        ELSE 0
    END AS "Tasa_Anual_100k"
FROM delitos_crudos d
LEFT JOIN poblacion_agrupada p ON d."Cve_Municipio" = p."Cve_Municipio";


--Creamos el DATA MART y los indices
CREATE TABLE dm_modelado_delitos AS
SELECT
    "Clave_Ent",
    "Entidad",
    "Cve_Municipio",
    "Municipio",
    "Bien_juridico_afectado",
    "Tipo_de_delito",
    "Subtipo_de_delito",
    "Modalidad",
    "POB_TOTAL",
    -- Características temporales
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
    -- Variables importantes
    "Total_Anual",
    "Tasa_Anual_100k"
FROM vw_delitos_nivelados;

CREATE INDEX idx_dm_municipio ON dm_modelado_delitos("Cve_Municipio");
CREATE INDEX idx_dm_delito ON dm_modelado_delitos("Tipo_de_delito");


