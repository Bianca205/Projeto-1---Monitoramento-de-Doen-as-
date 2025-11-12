# 1. Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Definir paths
# Caminho para a raiz do projeto (uma pasta acima de Scripts)
PROJECT_ROOT = Path(__file__).parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Cria pastas, se não existirem
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# 3. Carregar dados brutos
df_raw = pd.read_csv(RAW_DATA_PATH / "heart.csv")

print(RAW_DATA_PATH / "heart.csv")
print((RAW_DATA_PATH / "heart.csv").exists())

# 4. Criar cópia para trabalhar
df_clean = df_raw.copy()

print(f"\n=== INICIANDO LIMPEZA ===")
print("Shape antes da limpeza:", df_clean.shape)

# 5. Remover duplicatas EXATAS
df_clean = df_clean.drop_duplicates()
print("Shape após remover duplicatas exatas:", df_clean.shape)

# 6. Verificar valores ausentes
print(f"\nValores ausentes por coluna:")
print(df_clean.isnull().sum())

# 7. Verificar balanceamento da variável target
target_col = 'target'  # ajuste conforme seu dataset
print(f"\nBalanceamento da variável target:")
print(df_clean[target_col].value_counts(normalize=True))

# 8. Verificar consistência dos valores numéricos
def check_data_quality(df):
    print("\n=== ANÁLISE DE QUALIDADE ===")
    print("Valores únicos por coluna:")
    for col in df.columns:
        unique_vals = df[col].nunique()
        print(f"{col}: {unique_vals} valores únicos")
        
    print("\nEstatísticas descritivas:")
    print(df.describe())

check_data_quality(df_clean)

# 9. Verificar e tratar valores inconsistentes
def check_value_ranges(df):
    """Verificar se os valores estão em ranges médicos plausíveis"""
    
    # Idade (anos)
    if 'age' in df.columns:
        invalid_age = df[(df['age'] < 0) | (df['age'] > 120)]
        print(f"Valores de idade suspeitos: {len(invalid_age)}")
    
    # Pressão arterial em repouso (mm Hg)
    if 'trestbps' in df.columns:
        invalid_bp = df[(df['trestbps'] < 50) | (df['trestbps'] > 250)]
        print(f"Valores de pressão arterial suspeitos: {len(invalid_bp)}")
    
    # Colesterol sérico (mg/dl)
    if 'chol' in df.columns:
        invalid_chol = df[(df['chol'] < 50) | (df['chol'] > 600)]
        print(f"Valores de colesterol suspeitos: {len(invalid_chol)}")
    
    # Frequência cardíaca máxima alcançada
    if 'thalach' in df.columns:
        invalid_thalach = df[(df['thalach'] < 40) | (df['thalach'] > 220)]
        print(f"Valores de frequência cardíaca máxima suspeitos: {len(invalid_thalach)}")
    
    # Depressão de ST induzida por exercício
    if 'oldpeak' in df.columns:
        invalid_oldpeak = df[(df['oldpeak'] < 0) | (df['oldpeak'] > 10)]
        print(f"Valores de depressão ST suspeitos: {len(invalid_oldpeak)}")

check_value_ranges(df_clean)

# 10. Salvar dados limpos
def save_clean_data(df):
    """Salva os dados processados"""
    output_file = PROCESSED_DATA_PATH / "heart_cleaned.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Dados limpos salvos em: {output_file}")
    return output_file

clean_file_path = save_clean_data(df_clean)

# 11. Resumo final da limpeza
print(f"\n=== RESUMO DA LIMPEZA ===")
print(f"Dataset original: {df_raw.shape}")
print(f"Dataset limpo: {df_clean.shape}")
print(f"Duplicatas removidas: {len(df_raw) - len(df_clean)}")
print(f"Redução de: {((len(df_raw) - len(df_clean)) / len(df_raw)) * 100:.2f}%")

# 12. Preparar para próxima fase (ML)
print(f"\n=== PRÓXIMOS PASSOS ===")
print("1. Os dados limpos estão em: data/processed/heart_cleaned.csv")
print("2. Próxima fase: Feature engineering e modelagem")
print(f"3. Tamanho final do dataset: {df_clean.shape}")