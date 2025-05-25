import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r'C:\Users\andre\OneDrive\Área de Trabalho\Python\IA\cap15_arvores\loan_data.csv')

dados = pd.get_dummies(data=df,columns=['purpose'],drop_first=True) #transforma a coluna purpose em colunas numericas

X = dados.drop('not.fully.paid',axis=1)
Y = dados['not.fully.paid']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)

#------------Tree decision: cria a estrutura de decisao numa arvore so------
dtree = DecisionTreeClassifier()
dtree.fit(X_train,Y_train)
pred = dtree.predict(X_test)

cr = classification_report(Y_test,pred)
cmat = confusion_matrix(Y_test,pred)

print(cr)
print(cmat)

#----------random forest: cria varias arvores e depois faz a votaçao da decisao com elas

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,Y_train)
pred = rfc.predict(X_test)

cr = classification_report(Y_test,pred)
cmat = confusion_matrix(Y_test,pred)

print(cr)
print(cmat)

