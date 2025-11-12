# Passo 1: Carregar Dados Limpos e Configuração

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# Configurar paths
PROJECT_ROOT = Path(__file__).parent.parent
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
FINAL_FEATURES_PATH = PROCESSED_PATH / "heart_features.csv"

# Carregar dados limpos
df = pd.read_csv(PROCESSED_PATH / "heart_cleaned.csv")
print(f"Dataset carregado: {df.shape}")

# Passo 2: Análise das Colunas Existentes

# Verificar colunas disponíveis para engineering
print("Colunas disponíveis:")
print(df.columns.tolist())
print("\nTipos de dados:")
print(df.dtypes)
print("\nPrimeiras linhas:")
print(df.head())

#Passo 3: Feature Engineering - Criar Novas Variáveis

def create_new_features(df):
    """
    Criar novas features baseadas em conhecimento médico
    """
    df_eng = df.copy()
    
    # 1. IDADE: Criar faixas etárias 
    if 'age' in df.columns:
        df_eng['age_group'] = pd.cut(df_eng['age'], 
                                   bins=[0, 40, 50, 60, 70, 100],
                                   labels=['<40', '40-50', '50-60', '60-70', '70_mais'])
    
    # 2. PRESSÃO ARTERIAL: Classificar níveis 
    if 'trestbps' in df.columns:
        conditions = [
            df_eng['trestbps'] < 120,
            (df_eng['trestbps'] >= 120) & (df_eng['trestbps'] < 130),
            (df_eng['trestbps'] >= 130) & (df_eng['trestbps'] < 140),
            (df_eng['trestbps'] >= 140) & (df_eng['trestbps'] < 160),
            df_eng['trestbps'] >= 160
        ]
        categories = ['Normal', 'Elevated', 'Hypertension_Stage1', 'Hypertension_Stage2', 'Hypertension_Crisis']
        df_eng['bp_category'] = np.select(conditions, categories, default='Unknown')
    
    # 3. COLESTEROL: Classificar níveis 
    if 'chol' in df.columns:
        df_eng['chol_category'] = pd.cut(df_eng['chol'],
                                       bins=[0, 200, 240, 1000],
                                       labels=['Normal', 'Borderline', 'High'])
    
    # 4. FREQUÊNCIA CARDÍACA: Análise relativa (se 'thalach' e 'age' existirem)
    if all(col in df.columns for col in ['thalach', 'age']):
        # Frequência cardíaca máxima prevista (220 - idade)
        df_eng['max_hr_predicted'] = 220 - df_eng['age']
        df_eng['hr_percentage_max'] = (df_eng['thalach'] / df_eng['max_hr_predicted']) * 100
        
        # Categorizar desempenho cardíaco
        df_eng['hr_performance'] = pd.cut(df_eng['hr_percentage_max'],
                                        bins=[0, 60, 85, 100, 200],
                                        labels=['Below_Normal', 'Normal', 'Good', 'Excellent'])
    
    # 5. ÍNDICES COMPOSTOS (se colunas relevantes existirem)
    if all(col in df.columns for col in ['trestbps', 'chol', 'age']):
        # Score de risco simplificado (exemplo)
        df_eng['risk_score'] = (df_eng['trestbps'] / 100) + (df_eng['chol'] / 200) + (df_eng['age'] / 50)
    
    # 6. INTERAÇÕES ENTRE VARIÁVEIS
    if all(col in df.columns for col in ['sex', 'age']):
        # Idade por gênero (pode capturar diferenças de risco)
        df_eng['age_male'] = df_eng['age'] * df_eng['sex']
        df_eng['age_female'] = df_eng['age'] * (1 - df_eng['sex'])
    
    return df_eng

# Aplicar feature engineering
df_engineered = create_new_features(df)
print(f"Dataset após feature engineering: {df_engineered.shape}")


# Passo 4: Codificação de Variáveis Categóricas

def encode_categorical_features(df):
    """
    Codificar variáveis categóricas para formato numérico
    """
    df_encoded = df.copy()
    
    # Lista de colunas categóricas criadas
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Colunas categóricas para codificar: {categorical_cols}")
    
    # One-Hot Encoding para variáveis nominais
    nominal_cols = [col for col in categorical_cols if 'category' in col or 'group' in col or 'performance' in col]
    
    if nominal_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=nominal_cols, prefix=nominal_cols, drop_first=True)
    
    # Label Encoding para variáveis binárias (se houver)
    binary_cols = [col for col in categorical_cols if col not in nominal_cols and df_encoded[col].nunique() == 2]
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in binary_cols:
        df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
        df_encoded = df_encoded.drop(columns=[col])
    
    return df_encoded

df_final = encode_categorical_features(df_engineered)
print(f"Dataset após codificação: {df_final.shape}")


#Passo 5: Preparação Final para ML

def prepare_ml_dataframe(df):
    """
    Preparar DataFrame final para modelagem de ML
    """
    # 1. Identificar variável target (ajuste o nome conforme seu dataset)
    target_column = 'target'  # Substitua pelo nome real da sua coluna target
    
    if target_column not in df.columns:
        print("Colunas disponíveis:", df.columns.tolist())
        raise ValueError(f"Target column '{target_column}' não encontrada!")
    
    # 2. Separar features e target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 3. Remover colunas com baixa variância ou alta correlação
    # Remover colunas com apenas 1 valor único
    low_variance_cols = [col for col in X.columns if X[col].nunique() <= 1]
    if low_variance_cols:
        print(f"Removendo colunas com baixa variância: {low_variance_cols}")
        X = X.drop(columns=low_variance_cols)
    
    # 4. Verificar correlações (opcional - remover features muito correlacionadas)
    correlation_matrix = X.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    high_corr_cols = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
    
    if high_corr_cols:
        print(f"Colunas altamente correlacionadas: {high_corr_cols}")
        X = X.drop(columns=high_corr_cols)
    
    # 5. Juntar features e target novamente
    df_ml_ready = pd.concat([X, y], axis=1)
    
    return df_ml_ready, X, y

# Preparar dados para ML
df_ml_ready, X_features, y_target = prepare_ml_dataframe(df_final)
print(f"Dataset final para ML: {df_ml_ready.shape}")
print(f"Features: {X_features.shape}")
print(f"Target: {y_target.shape}")


#Passo 6: Análise das Features Criadas

def analyze_final_features(df, X, y):
    """
    Análise final do dataset preparado
    """
    print("\n=== ANÁLISE FINAL DO DATASET ===")
    print(f"Shape final: {df.shape}")
    print(f"Número de features: {X.shape[1]}")
    print(f"Distribuição do target:")
    print(y.value_counts(normalize=True))
    
    # Listar todas as features
    print(f"\nFeatures disponíveis ({len(X.columns)}):")
    for i, col in enumerate(X.columns, 1):
        print(f"{i:2d}. {col}")
    
    # Correlação com target
    correlation_with_target = pd.concat([X, y], axis=1).corr()[y.name].sort_values(ascending=False)
    print(f"\nTop 10 features mais correlacionadas com target:")
    print(correlation_with_target.head(10))

analyze_final_features(df_ml_ready, X_features, y_target)


# Passo 7: Salvar Dataset Final

def save_final_datasets(df_ml_ready, X_features, y_target):
    """
    Salvar datasets finais para modelagem
    """
    # Salvar dataset completo
    df_ml_ready.to_csv(FINAL_FEATURES_PATH, index=False)
    
    # Salvar features e target separados (útil para modelagem)
    X_features.to_csv(PROCESSED_PATH / "X_features.csv", index=False)
    y_target.to_csv(PROCESSED_PATH / "y_target.csv", index=False)
    
    # Salvar lista de features
    feature_list = X_features.columns.tolist()
    pd.Series(feature_list).to_csv(PROCESSED_PATH / "feature_list.csv", index=False, header=['feature_name'])
    
    print(f"\n=== DATASETS SALVOS ===")
    print(f"Dataset completo: {FINAL_FEATURES_PATH}")
    print(f"Features (X): data/processed/X_features.csv")
    print(f"Target (y): data/processed/y_target.csv")
    print(f"Lista de features: data/processed/feature_list.csv")

save_final_datasets(df_ml_ready, X_features, y_target)