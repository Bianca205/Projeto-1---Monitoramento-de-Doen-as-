# 🩺 Monitoramento de Saúde e Predição de Doenças

Projeto de Ciência de Dados e Machine Learning voltado ao **monitoramento de saúde** e à **predição de risco de doenças** (como diabetes e doenças cardíacas) a partir de dados clínicos, utilizando um pipeline completo de dados, banco relacional, modelos de classificação e dashboard interativo.

---

## 🎯 Objetivo

Desenvolver uma solução fim-a-fim que:
- Organize dados médicos em um **banco de dados SQL**
- Limpe e transforme exames clínicos via **pipeline de dados**
- Utilize **Machine Learning** para prever o risco de doenças
- Disponibilize os resultados em um **dashboard interativo**

---

## 📊 Dados

- Origem: **Kaggle**
- Exemplos de datasets:
  - Diabetes (Pima Indians Diabetes Dataset)
  - Doenças cardíacas (Heart Disease Dataset – UCI/Kaggle)
- Tipo de dados:
  - Idade
  - Sexo
  - Indicadores clínicos (glicose, IMC, pressão arterial, colesterol, etc.)
  - Diagnóstico (label binário: risco / não risco)

> ⚠️ Os dados são utilizados **apenas para fins educacionais**, sem qualquer relação com pacientes reais.

---

## 🗄️ Modelagem de Dados (SQL)

O banco foi estruturado de forma genérica para suportar diferentes exames e doenças.

### Principais tabelas:
- **patients**: informações demográficas
- **exams**: registros de exames realizados
- **exam_results**: valores individuais dos exames
- **diagnoses**: diagnósticos reais (labels)
- **predictions**: resultados previstos pelo modelo

Essa modelagem permite reutilização do sistema para múltiplos datasets e condições clínicas.

---

## 🔄 Pipeline de Dados (ETL)

Fluxo implementado:

1. **Ingestão**
   - Leitura de datasets brutos (CSV)
2. **Limpeza**
   - Remoção de duplicatas
   - Tratamento de valores ausentes
   - Correção de valores inconsistentes
3. **Transformação**
   - Padronização de colunas
   - Normalização/escala de features
4. **Carga**
   - Inserção dos dados tratados no banco SQL
5. **Feature Engineering**
   - Seleção de variáveis relevantes para o modelo

---

## 🤖 Machine Learning

- Tipo: **Classificação**
- Modelos utilizados:
  - Logistic Regression (baseline)
  - Random Forest / Gradient Boosting
- Métricas avaliadas:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
- Saída:
  - Probabilidade de risco
  - Classe prevista (0 = baixo risco, 1 = alto risco)

> O foco do projeto é **interpretação e confiabilidade**, não apenas acurácia.

---

## 📈 Dashboard

Dashboard interativo desenvolvido com **Streamlit** (ou Power BI, como alternativa), contendo:

- Visão geral das métricas do modelo
- Simulação de predição a partir de dados de entrada
- Visualização da importância das variáveis
- Distribuição dos dados clínicos


---

## 🛠️ Tecnologias Utilizadas

- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **SQL (PostgreSQL / SQLite)**
- **Streamlit**
- **Matplotlib / Seaborn**
- **Git & GitHub**

---

## ▶️ Como Executar o Projeto

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/health-risk-monitoring.git
pip install -r requirements.txt
Execute o pipeline de dados
python src/train/train_model.py
streamlit run app/main.py
```
### 🎥 Demonstração

#### 📺 Vídeo demonstrando o funcionamento do sistema (Bubble / Streamlit / Pipeline):

#### 👉 link do vídeo aqui: [Monitoramento de Saúde - Predição de Doenças Cardíacas](https://drive.google.com/drive/folders/18Llqn3j92AeeMABqkX8ohguwSS0h--oo?usp=sharing)

### 📌 Observações Importantes

- Este projeto não substitui diagnóstico médico

- Desenvolvido exclusivamente para fins acadêmicos e de portfólio

- Pode ser facilmente estendido para:
  - novas doenças
  - novos datasets
  - integração com APIs

### 👩‍💻 Autoria

Projeto desenvolvido por: Gyovanna Garcês



