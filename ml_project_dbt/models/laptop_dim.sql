{{ config(materialized='table') }}

SELECT 

CompanyName as company_name,
TypeOfLaptop as type_of_laptop,
Inches as inches,
ScreenResolution as screen_resolution,
Cpu as cpu,
Ram as ram,
Memory as memory,
Gpu as gpu,
OpSys as os,
TRUNC(Weight, 2) as weight,
Price::INTEGER as price,

FROM laptop_raw
