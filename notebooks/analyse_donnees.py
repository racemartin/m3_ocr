import pandas as pd
import polars as pl
import os
import polars as pl
import pandas as pd
import os

def charger_et_analyser_donnees(filepath, engine="polars"):
    print("\n" + "=" * 100)
    print(f"ÉTAPE 0 : ANALYSE EXPLORATOIRE INITIALE ({engine.upper()})")
    print("=" * 100)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier n'a pas été trouvé : {filepath}")

    extension = os.path.splitext(filepath)[1].lower()
    
    if engine == "polars":
        df = pl.read_csv(filepath, infer_schema_length=10000, ignore_errors=True)
        rows, cols = df.shape
        
        # --- CÁLCULO DE MÉTRICAS ENRIQUECIDAS ---
        summary_data = []
        for col in df.columns:
            dtype = df.schema[col]
            
            # 1. Determinar Naturaleza
            if dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                nature = "Numérique"
            elif dtype == pl.Boolean:
                nature = "Booléen"
            else:
                nature = "Qualitative"
                
            # 2. Métricas
            non_null = df.select(pl.col(col).drop_nulls().count()).item()
            nans = rows - non_null
            cardinality = df.select(pl.col(col).n_unique()).item()
            
            # 3. Categoría de Cardinalidad (Lógica simple)
            if cardinality == 1: cat = "CONSTANTE"
            elif cardinality < 10: cat = "BASSE"
            elif cardinality < 50: cat = "MODÉRÉE"
            else: cat = "HAUTE"

            summary_data.append({
                "Columna": col,
                "Tipo": str(dtype),
                "Naturaleza": nature,
                "Cardinalidad": cardinality,
                "No Nulos": non_null,
                "NaN": nans,
                "Categoría": cat
            })
        
        analysis_df = pd.DataFrame(summary_data) # Usamos pandas solo para el print bonito en tabla
        
    else: # Versión simplificada para Pandas si fuera necesario
        df = pd.read_csv(filepath)
        rows, cols = df.shape
        # (Aquí podrías replicar la lógica, pero Polars es el motor principal)

    print(f"\n📊 Dimensions : {rows:,} lignes x {cols} colonnes")
    print(f"\n📋 ANALYSE DÉTAILLÉE DES FEATURES :")
    print(analysis_df.to_markdown(index=False))
    
    return df