--VISTA MATERIALIZADA
DROP MATERIALIZED VIEW IF EXISTS vw_municipios_pivote CASCADE;

CREATE MATERIALIZED VIEW vw_municipios_pivote AS
WITH poblacion_agrupada AS (
    -- Una sola fila de población por municipio
    SELECT 
        "Cve_Municipio",
        SUM("POB_TOTAL") AS poblacion_total
    FROM poblacion_cruda
    GROUP BY "Cve_Municipio"
)
SELECT 
    d."Cve_Municipio",
    d."Municipio",
    COALESCE(p.poblacion_total, 0) AS "Poblacion_Total",

   --Delitos que se desplegarán como columnas
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Homicidio'), 0) AS "Homicidio",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Lesiones'), 0) AS "Lesiones",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Feminicidio'), 0) AS "Feminicidio",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Aborto'), 0) AS "Aborto",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE 'Otros delitos que atentan contra la vida%'), 0) AS "Otros_Vida",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Secuestro'), 0) AS "Secuestro",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Tráfico de menores'), 0) AS "Trafico_Menores",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Rapto'), 0) AS "Rapto",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE 'Otros delitos que atentan contra la libertad personal%'), 0) AS "Otros_Libertad",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Abuso sexual'), 0) AS "Abuso_Sexual",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Acoso sexual'), 0) AS "Acoso_Sexual",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Hostigamiento sexual'), 0) AS "Hostigamiento_Sexual",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Violación simple'), 0) AS "Violacion_Simple",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Violación equiparada'), 0) AS "Violacion_Equiparada",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Incesto'), 0) AS "Incesto",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE 'Otros delitos que atentan contra la libertad y la seguridad sexual%'), 0) AS "Otros_Sexual",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE '%Robo%'), 0) AS "Robos_Totales",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Fraude'), 0) AS "Fraude",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Abuso de confianza'), 0) AS "Abuso_Confianza",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Extorsión'), 0) AS "Extorsion",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Daño a la propiedad'), 0) AS "Dano_Propiedad",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Despojo'), 0) AS "Despojo",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE 'Otros delitos contra el patrimonio%'), 0) AS "Otros_Patrimonio",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Violencia familiar'), 0) AS "Violencia_Familiar",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE '%Violencia de género%'), 0) AS "Violencia_Genero",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE '%Incumplimiento de obligaciones%'), 0) AS "Incumplimiento_Obligaciones",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE 'Otros delitos contra la familia%'), 0) AS "Otros_Familia",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Corrupción de menores'), 0) AS "Corrupcion_Menores",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Trata de personas'), 0) AS "Trata_Personas",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE 'Otros delitos contra la sociedad%'), 0) AS "Otros_Sociedad",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Narcomenudeo'), 0) AS "Narcomenudeo",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Amenazas'), 0) AS "Amenazas",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Allanamiento de morada'), 0) AS "Allanamiento_Morada",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Evasión de presos'), 0) AS "Evasion_Presos",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Falsedad'), 0) AS "Falsedad",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Falsificación'), 0) AS "Falsificacion",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Contra el medio ambiente'), 0) AS "Medio_Ambiente",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Delitos cometidos por servidores públicos'), 0) AS "Servidores_Publicos",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" = 'Electorales'), 0) AS "Electorales",
    COALESCE(SUM(d."Total_Anual") FILTER (WHERE d."Tipo_de_delito" ILIKE 'Otros delitos del Fuero Común%'), 0) AS "Otros_Fuero_Comun",
    
    -- Suma total de todos los delitos
    SUM(d."Total_Anual") AS "Total_Delitos_Absoluto"

FROM delitos_crudos d
LEFT JOIN poblacion_agrupada p ON d."Cve_Municipio" = p."Cve_Municipio"
-- Agrupamos por la llave del municipio para que cada municipio sea una única fila
GROUP BY 
    d."Cve_Municipio", 
    d."Municipio", 
    p.poblacion_total;

--DATA MART FÍSICO
DROP TABLE IF EXISTS dm_municipios_features CASCADE;

CREATE TABLE dm_municipios_features AS
SELECT 
    *,
    --Tasa Global por cada 100k habitantes
    CASE 
        WHEN "Poblacion_Total" > 0 THEN 
            ROUND(("Total_Delitos_Absoluto"::numeric / "Poblacion_Total") * 100000, 2)
        ELSE 0 
    END AS "Tasa_Global_100k"
FROM vw_municipios_pivote;

-- CREAR ÍNDICE
CREATE UNIQUE INDEX idx_dm_municipio_pivote ON dm_municipios_features("Cve_Municipio");
