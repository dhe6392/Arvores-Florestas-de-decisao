# Árvores e Florestas de Decisão

Este projeto compara dois modelos de classificação supervisionada aplicados a um conjunto de dados de empréstimos (`loan_data.csv`):

- **Árvore de Decisão (Decision Tree)**
- **Random Forest**

O objetivo é prever se um cliente **não pagará totalmente** um empréstimo (`not.fully.paid`), com base em variáveis como histórico de crédito, renda e finalidade do empréstimo.

## 📊 Dados
O dataset é processado com `get_dummies` para transformar a variável categórica `purpose` em variáveis numéricas (one-hot encoding). 
Obs: Os dados da planilha não necessariamente são reais.

## 📌 Etapas do projeto

1. Carregamento e pré-processamento dos dados
2. Treinamento de uma **Árvore de Decisão**
3. Avaliação com `classification_report` e `confusion_matrix`
4. Treinamento de um modelo **Random Forest** com 600 árvores
5. Comparação de desempenho entre os dois modelos

## 📈 Resultados

A acurácia da árvore de decisão foi de **73%**, enquanto a random forest atingiu **84%**. Apesar disso, ambos os modelos tiveram dificuldade para prever corretamente os casos de inadimplência, devido ao desbalanceamento das classes.

## 📚 Bibliotecas usadas

- `pandas`
- `sklearn.tree.DecisionTreeClassifier`
- `sklearn.ensemble.RandomForestClassifier`
- `sklearn.metrics`
- `train_test_split`


