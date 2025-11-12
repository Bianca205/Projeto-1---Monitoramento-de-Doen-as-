# Passo 1: Configuração e Importações
# model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# Passo 2: Carregar Dados Processados
def load_processed_data():
    """Carregar dados após feature engineering"""
    PROJECT_ROOT = Path(__file__).parent.parent
    PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
    
    X = pd.read_csv(PROCESSED_DATA_PATH / "X_features.csv")
    y = pd.read_csv(PROCESSED_DATA_PATH / "y_target.csv")
    
    # Se y for DataFrame com uma coluna, converter para Series
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]
    
    print(f"✅ Dados carregados: X {X.shape}, y {y.shape}")
    return X, y

# Passo 3: Divisão Treino/Teste e Escalonamento
def prepare_train_test(X, y, test_size=0.2, random_state=42):
    """Dividir e escalonar dados"""
    
    # Divisão estratificada (mantém proporção do target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Divisão treino/teste:")
    print(f"   Treino: {X_train.shape}, {y_train.shape}")
    print(f"   Teste:  {X_test.shape}, {y_test.shape}")
    
    # Escalonamento (importante para alguns modelos)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Salvar scaler para uso futuro
    joblib.dump(scaler, "models/scaler.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Passo 4: Definir Modelos para 

def get_models():
    """Definir modelos para comparação"""
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(random_state=42, probability=True),
    }
    
    return models

# Passo 5: Treinamento e Avaliação dos Modelos

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Treinar e avaliar múltiplos modelos"""
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🎯 Treinando {name}...")
        
        # Treinar modelo
        model.fit(X_train, y_train)
        
        # Previsões
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Validação cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Armazenar resultados
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'auc': auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"   ✅ Acurácia: {accuracy:.4f}")
        print(f"   ✅ AUC: {auc:.4f}")
        print(f"   ✅ CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results

#Passo 6: Comparação e Seleção do Melhor Modelo
def select_best_model(results):
    """Selecionar o melhor modelo baseado nas métricas"""
    
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[name]['accuracy'] for name in results.keys()],
        'AUC': [results[name]['auc'] for name in results.keys()],
        'CV_Mean': [results[name]['cv_mean'] for name in results.keys()],
        'CV_Std': [results[name]['cv_std'] for name in results.keys()]
    }).sort_values('AUC', ascending=False)
    
    print("\n🏆 COMPARAÇÃO DE MODELOS:")
    print(results_df.round(4))
    
    # Selecionar melhor modelo (maior AUC)
    best_model_name = results_df.iloc[0]['Model']
    best_model = results[best_model_name]['model']
    
    print(f"\n⭐ MELHOR MODELO: {best_model_name}")
    print(f"📈 AUC: {results_df.iloc[0]['AUC']:.4f}")
    print(f"🎯 Acurácia: {results_df.iloc[0]['Accuracy']:.4f}")
    
    return best_model_name, best_model, results[best_model_name]

#Passo 7: Otimização de Hiperparâmetros (Opcional)


def optimize_hyperparameters(best_model_name, X_train, y_train):
    """Otimizar hiperparâmetros do melhor modelo"""
    
    if best_model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    elif best_model_name == 'Logistic Regression':
        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs']
        }
        model = LogisticRegression(random_state=42)
    
    else:
        print("⏭️  Pulando otimização para este modelo")
        return None
    
    print(f"\n🔧 Otimizando {best_model_name}...")
    
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Melhores parâmetros: {grid_search.best_params_}")
    print(f"✅ Melhor score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

#Passo 8: Salvar Modelo e Resultados
def save_model_and_results(best_model, best_results, model_name, X):
    """Salvar modelo treinado e resultados"""
    
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    
    # Salvar modelo
    model_filename = f"best_model_{model_name.replace(' ', '_').lower()}.pkl"
    joblib.dump(best_model, models_path / model_filename)
    
    # Salvar métricas
    metrics = {
        'model_name': model_name,
        'accuracy': best_results['accuracy'],
        'auc': best_results['auc'],
        'cv_score': best_results['cv_mean'],
        'features_used': X.shape[1]
    }
    
    pd.DataFrame([metrics]).to_csv(models_path / "model_metrics.csv", index=False)
    
    print(f"\n💾 Modelo salvo: models/{model_filename}")
    print(f"💾 Métricas salvas: models/model_metrics.csv")
    
    return model_filename

# Passo 9: Visualização dos Resultados
def plot_results(results, best_model_name):
    """Criar visualizações dos resultados"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfico de comparação de modelos
    models = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in models]
    auc_scores = [results[name]['auc'] for name in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, accuracies, width, label='Acurácia', alpha=0.7)
    axes[0].bar(x + width/2, auc_scores, width, label='AUC', alpha=0.7)
    axes[0].set_xlabel('Modelos')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Comparação de Modelos')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Matriz de confusão do melhor modelo
    cm = results[best_model_name]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
    axes[1].set_title(f'Matriz de Confusão - {best_model_name}')
    axes[1].set_xlabel('Predito')
    axes[1].set_ylabel('Real')
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

#Passo 10: Função Principal
def main():
    """Função principal"""
    print("🚀 INICIANDO TREINAMENTO DE MODELOS...")
    
    # 1. Carregar dados
    X, y = load_processed_data()
    
    # 2. Preparar dados
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(X, y)
    
    # 3. Obter modelos
    models = get_models()
    
    # 4. Treinar e avaliar
    results = evaluate_models(models, X_train, X_test, y_train, y_test)
    
    # 5. Selecionar melhor
    best_name, best_model, best_results = select_best_model(results)
    
    # 6. Otimizar (opcional)
    optimized_model = optimize_hyperparameters(best_name, X_train, y_train)
    if optimized_model:
        best_model = optimized_model
    
    # 7. Salvar
    model_filename = save_model_and_results(best_model, best_results, best_name, X)
    
    # 8. Visualizar
    plot_results(results, best_name)
    
    print(f"\n🎉 TREINAMENTO CONCLUÍDO!")
    print(f"📊 Melhor modelo: {best_name}")
    print(f"💾 Arquivo: {model_filename}")

if __name__ == "__main__":
    main()