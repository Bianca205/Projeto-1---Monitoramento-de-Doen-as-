# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sklearn
import sys

# Configuração da página
st.set_page_config(
    page_title="Sistema de Predição de Doenças Cardíacas",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e63946;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #e63946;
    }
    .prediction-high {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e63946;
    }
    .prediction-low {
        background-color: #e6f7e6;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">❤️ Monitoramento de Saúde - Predição de Doenças Cardíacas</h1>', unsafe_allow_html=True)

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.radio("Selecione a página:", 
                       ["📊 Dashboard Geral", "🎯 Predição em Tempo Real", "📈 Análise do Modelo", "ℹ️ Sobre o Projeto"])

# Carregar modelo e dados
@st.cache_resource
def load_model():
    """Carregar modelo treinado"""
    try:
        model_path = Path("models") / "best_model_svm.pkl"  # Ajuste conforme seu modelo
        scaler_path = Path("models") / "scaler.pkl"        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        return model, scaler
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, None

@st.cache_data
def load_data():
    """Carregar dados processados"""
    try:
        data_path = Path("data/processed/heart_features.csv")
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None

@st.cache_data
def load_metrics():
    """Carregar métricas do modelo"""
    try:
        metrics_path = Path("models") / "model_metrics.csv"
        metrics = pd.read_csv(metrics_path)
        return metrics.iloc[0]  # Retorna a primeira linha como Series
    except:
        return None

# Carregar recursos
model, scaler = load_model()
df = load_data()
metrics = load_metrics()

# Página 1: Dashboard Geral
if page == "📊 Dashboard Geral":
    st.header("📊 Dashboard Analítico")
    
    if df is not None:
        # Métricas rápidas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Pacientes", len(df))
        
        with col2:
            positive_cases = df[df['target'] == 1].shape[0] if 'target' in df.columns else 0
            st.metric("Casos Positivos", positive_cases)
        
        with col3:
            negative_cases = df[df['target'] == 0].shape[0] if 'target' in df.columns else 0
            st.metric("Casos Negativos", negative_cases)
        
        with col4:
            prevalence = (positive_cases / len(df)) * 100 if len(df) > 0 else 0
            st.metric("Prevalência", f"{prevalence:.1f}%")
        
        # Gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribuição da Variável Target")
            if 'target' in df.columns:
                target_counts = df['target'].value_counts()
                fig = px.pie(values=target_counts.values, 
                           names=['Sem Doença', 'Com Doença'],
                           color=['Sem Doença', 'Com Doença'],
                           color_discrete_map={'Sem Doença':'lightblue', 'Com Doença':'lightcoral'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Distribuição de Idade")
            if 'age' in df.columns:
                fig = px.histogram(df, x='age', nbins=20, 
                                 color_discrete_sequence=['#e63946'])
                fig.update_layout(xaxis_title="Idade", yaxis_title="Contagem")
                st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap de correlação
        st.subheader("Mapa de Calor - Correlações")
        if len(df.columns) > 1:
            # Selecionar apenas colunas numéricas
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                
                fig = px.imshow(corr_matrix,
                              color_continuous_scale='RdBu_r',
                              aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Não foi possível carregar os dados para análise.")

# Página 2: Predição em Tempo Real
elif page == "🎯 Predição em Tempo Real":
    st.header("🎯 Predição de Risco em Tempo Real")
    
    if model is not None and scaler is not None:
        st.info("Preencha os dados do paciente para obter uma predição de risco")
        
        # Formulário de entrada - dividido em colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dados Demográficos")
            age = st.slider("Idade", 20, 100, 50)
            sex = st.selectbox("Sexo", options=[0, 1], 
                             format_func=lambda x: "Feminino" if x == 0 else "Masculino")
        
        with col2:
            st.subheader("Medições Clínicas")
            trestbps = st.slider("Pressão Arterial em Repouso (mm Hg)", 80, 200, 120)
            chol = st.slider("Colesterol Sérico (mg/dl)", 100, 600, 200)
            thalach = st.slider("Frequência Cardíaca Máxima", 60, 220, 150)
            oldpeak = st.slider("Depressão do ST", 0.0, 6.0, 1.0)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Sintomas")
            cp = st.selectbox("Tipo de Dor Torácica", 
                            options=[0, 1, 2, 3],
                            format_func=lambda x: ["Angina típica", "Angina atípica", 
                                                 "Dor não anginal", "Assintomático"][x])
            exang = st.selectbox("Angina Induzida por Exercício", 
                               options=[0, 1],
                               format_func=lambda x: "Não" if x == 0 else "Sim")
        
        with col4:
            st.subheader("Exames")
            fbs = st.selectbox("Açúcar no Sangue em Jejum > 120 mg/dl",
                             options=[0, 1],
                             format_func=lambda x: "Não" if x == 0 else "Sim")
            restecg = st.selectbox("Resultado Eletrocardiográfico em Repouso",
                                 options=[0, 1, 2],
                                 format_func=lambda x: ["Normal", "Anormal", "Hipertrofia"][x])
            slope = st.selectbox("Inclinação do Segmento ST",
                               options=[0, 1, 2],
                               format_func=lambda x: ["Ascendente", "Plana", "Descendente"][x])
            ca = st.slider("Número de Vasos Principais", 0, 3, 1)
            thal = st.selectbox("Thalassemia",
                              options=[0, 1, 2, 3],
                              format_func=lambda x: ["Normal", "Defeito Fixo", 
                                                   "Defeito Reversível", "Outro"][x])
        
        # Botão de predição
        if st.button("🔍 Realizar Predição", type="primary"):
            
            # FUNÇÃO PARA APLICAR O MESMO FEATURE ENGINEERING DO TREINAMENTO
            def apply_feature_engineering(input_dict):
                # Criar DataFrame
                input_df = pd.DataFrame([input_dict])
                
                # 1. IDADE: Criar faixas etárias 
                input_df['age_group'] = pd.cut(input_df['age'], 
                                            bins=[0, 40, 50, 60, 70, 100],
                                            labels=['<40', '40-50', '50-60', '60-70', '70_mais'])
                
                # 2. PRESSÃO ARTERIAL: Classificar níveis
                conditions = [
                    input_df['trestbps'] < 120,
                    (input_df['trestbps'] >= 120) & (input_df['trestbps'] < 130),
                    (input_df['trestbps'] >= 130) & (input_df['trestbps'] < 140),
                    (input_df['trestbps'] >= 140) & (input_df['trestbps'] < 160),
                    input_df['trestbps'] >= 160
                ]
                categories = ['Normal', 'Elevated', 'Hypertension_Stage1', 'Hypertension_Stage2', 'Hypertension_Crisis']
                input_df['bp_category'] = np.select(conditions, categories, default='Unknown')
                
                # 3. COLESTEROL: Classificar níveis
                input_df['chol_category'] = pd.cut(input_df['chol'],
                                                bins=[0, 200, 240, 1000],
                                                labels=['Normal', 'Borderline', 'High'])
                
                # 4. FREQUÊNCIA CARDÍACA: Análise relativa
                input_df['max_hr_predicted'] = 220 - input_df['age']
                input_df['hr_percentage_max'] = (input_df['thalach'] / input_df['max_hr_predicted']) * 100
                
                input_df['hr_performance'] = pd.cut(input_df['hr_percentage_max'],
                                                  bins=[0, 60, 85, 100, 200],
                                                  labels=['Below_Normal', 'Normal', 'Good', 'Excellent'])
                
                # 5. SCORE DE RISCO
                input_df['risk_score'] = (input_df['trestbps'] / 100) + (input_df['chol'] / 200) + (input_df['age'] / 50)
                
                return input_df
            
            # FUNÇÃO PARA APLICAR ONE-HOT ENCODING
            def apply_onehot_encoding(df):
                df_encoded = df.copy()
                
                # Lista de colunas categóricas
                categorical_cols = ['age_group', 'bp_category', 'chol_category', 'hr_performance']
                
                # Aplicar one-hot encoding
                for col in categorical_cols:
                    if col in df_encoded.columns:
                        dummies = pd.get_dummies(df_encoded[col], prefix=col)
                        df_encoded = pd.concat([df_encoded, dummies], axis=1)
                        df_encoded = df_encoded.drop(columns=[col])
                
                return df_encoded
            
            # 1. Montar input básico
            input_dict = {
                'age': age,
                'sex': sex,
                'cp': cp,
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'restecg': restecg,
                'thalach': thalach,
                'exang': exang,
                'oldpeak': oldpeak,
                'slope': slope,
                'ca': ca,
                'thal': thal
            }
            
            # DEBUG: Mostrar input original
            st.subheader("🔍 DEBUG - Dados de Entrada Originais")
            st.dataframe(pd.DataFrame([input_dict]))
            
            # 2. Aplicar feature engineering (MESMO PROCESSO DO TREINAMENTO)
            input_engineered = apply_feature_engineering(input_dict)
            
            # DEBUG: Mostrar após feature engineering
            st.subheader("🔍 DEBUG - Após Feature Engineering")
            st.dataframe(input_engineered)
            
            # 3. Aplicar one-hot encoding
            input_encoded = apply_onehot_encoding(input_engineered)
            

            # 2. Carregar lista de features finais
            try:
                feature_list_path = Path(__file__).parent.parent / "data" / "processed" / "feature_list.csv"
                expected_features = pd.read_csv(feature_list_path)['feature_name'].tolist()
    
             # 5. Garantir que temos todas as features esperadas
                for feature in expected_features:
                    if feature not in input_encoded.columns:
                        input_encoded[feature] = 0  # Adicionar feature faltante com valor 0
                
                # 6. Manter apenas as features esperadas e na ordem correta
                input_final = input_encoded[expected_features]
                
                # DEBUG: Mostrar features finais
                st.subheader("🔍 DEBUG - Features Finais para o Modelo")
                st.write(f"Shape: {input_final.shape}")
                st.dataframe(input_final)
                
            except Exception as e:
                st.error(f"Erro ao carregar lista de features: {e}")
                st.stop()  # Use st.stop() em vez de return no Streamlit
            
            # 7. Escalonar dados (MESMO SCALER DO TREINAMENTO)
            input_scaled = scaler.transform(input_final)
            
            # 8. Fazer predição
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # DEBUG: Mostrar resultados intermediários
            st.subheader("🔍 DEBUG - Resultados Intermediários")
            col_debug1, col_debug2 = st.columns(2)
            
            with col_debug1:
                st.write("Dados escalonados (primeiras 10 features):")
                st.write(input_scaled[0][:10])
            
            with col_debug2:
                st.write("Probabilidades:")
                st.write(f"Classe 0 (Baixo Risco): {probability[0]:.3f}")
                st.write(f"Classe 1 (Alto Risco): {probability[1]:.3f}")
            
            # 9. Exibir resultados finais
            st.markdown("---")
            st.subheader("📋 Resultado da Predição")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.markdown('<div class="prediction-high">', unsafe_allow_html=True)
                    st.error("🩺 **ALTO RISCO** de Doença Cardíaca")
                    st.write(f"Probabilidade: {probability[1]:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-low">', unsafe_allow_html=True)
                    st.success("✅ **BAIXO RISCO** de Doença Cardíaca")
                    st.write(f"Probabilidade: {probability[0]:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            with col_result2:
                # Gráfico de probabilidade
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=['Baixo Risco', 'Alto Risco'],
                    y=[probability[0], probability[1]],
                    marker_color=['lightgreen', 'lightcoral']
                ))
                fig.update_layout(
                    title="Probabilidades de Predição",
                    yaxis_title="Probabilidade",
                    yaxis=dict(range=[0, 1])
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 10. Recomendações
            st.subheader("💡 Recomendações")
            if prediction == 1:
                st.warning("""
                **Recomendações para Alto Risco:**
                - Consulte um cardiologista urgentemente
                - Realize exames complementares
                - Monitore pressão arterial regularmente
                - Adote dieta saudável e pratique exercícios
                - Evite fumo e reduza consumo de álcool
                """)
            else:
                st.info("""
                **Manutenção da Saúde Cardíaca:**
                - Continue com check-ups regulares
                - Mantenha estilo de vida saudável
                - Pratique exercícios regularmente
                - Controle peso e alimentação
                - Evite fatores de risco
                """)
    
    else:
        st.error("Modelo não carregado. Verifique se o treinamento foi realizado corretamente.")

# Página 3: Análise do Modelo
elif page == "📈 Análise do Modelo":
    st.header("📈 Análise de Desempenho do Modelo")
    
    if metrics is not None:
        # Métricas do modelo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Acurácia", f"{metrics.get('accuracy', 0):.1%}")
        
        with col2:
            st.metric("AUC Score", f"{metrics.get('auc', 0):.1%}")
        
        with col3:
            st.metric("Validação Cruzada", f"{metrics.get('cv_score', 0):.1%}")
        
        with col4:
            st.metric("Número de Features", int(metrics.get('features_used', 0)))
        
        # Explicação do modelo
        st.subheader("📖 Sobre o Modelo")
        st.write(f"""
        **Modelo Utilizado:** {metrics.get('model_name', 'Random Forest')}
        
        Este modelo foi treinado para prever o risco de doenças cardíacas com base em 
        características clínicas e demográficas dos pacientes. A alta acurácia e AUC 
        indicam boa capacidade de distinguir entre pacientes de alto e baixo risco.
        """)
        
        # Feature Importance (se disponível)
        if model is not None and hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Importância das Features")
            
            # Carregar nomes das features
            try:
                X_features = pd.read_csv("data/processed/X_features.csv")
                feature_names = X_features.columns
                
                # Obter importâncias
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Criar gráfico
                fig = px.bar(x=importances[indices][:10], 
                           y=feature_names[indices][:10],
                           orientation='h',
                           title="Top 10 Features Mais Importantes",
                           labels={'x': 'Importância', 'y': 'Feature'})
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Não foi possível carregar informações de feature importance: {e}")
    
    else:
        st.warning("Métricas do modelo não disponíveis. Execute o treinamento primeiro.")

# Página 4: Sobre o Projeto
else:
    st.header("ℹ️ Sobre o Projeto")
    
    st.markdown("""
    ### 🎯 Objetivo
    Desenvolver um sistema de predição de doenças cardíacas utilizando Machine Learning
    para auxiliar profissionais de saúde na identificação precoce de riscos.
    
    ### 🛠️ Tecnologias Utilizadas
    - **Python** para análise e modelagem
    - **Scikit-learn** para algoritmos de ML
    - **Streamlit** para dashboard interativo
    - **Pandas & NumPy** para manipulação de dados
    - **Plotly & Matplotlib** para visualizações
    
    ### 📊 Dados
    - **Fonte:** Dataset de Doenças Cardíacas do Kaggle
    - **Amostras:** 302 pacientes após limpeza
    - **Features:** 13 características clínicas
    - **Target:** Presença (1) ou ausência (0) de doença cardíaca
    
    ### 🔬 Metodologia
    1. **Coleta e Limpeza de Dados**
    2. **Análise Exploratória (EDA)**
    3. **Feature Engineering**
    4. **Treinamento de Modelos**
    5. **Validação e Otimização**
    6. **Dashboard Interativo**
    
    ### 👥 Desenvolvido por
    Projeto desenvolvido para demonstração de capacidades em Ciência de Dados e ML.
    """)
    
    # Informações técnicas
    with st.expander("📋 Informações Técnicas Detalhadas"):
        st.write(f"**Versão do Python:** {sys.version}")
        st.write(f"**Versão do Scikit-learn:** {sklearn.__version__}")
        st.write(f"**Versão do Streamlit:** {st.__version__}")

# Rodapé
st.markdown("---")
st.markdown(
    "Desenvolvido com ❤️ usando Streamlit | "
    "⚠️ **Aviso:** Este é um projeto demonstrativo para fins educacionais."
)