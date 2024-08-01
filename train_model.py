import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
import joblib

def train_model():
    input_file = 'data/transformed_titanic_data.csv'
    model_file = 'data/titanic_survival_model.pkl'
    
    df = pd.read_csv(input_file)
    
    # Tratamento de valores ausentes
    df['Age'].fillna(df['Age'].median(), inplace=True)  # Preenche valores de idade faltantes com a mediana
    
    # Feature Engineering e seleção de variáveis
    X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    y = df['Survived']
    
    # Preprocessamento dos dados
    numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
    categorical_features = ['Pclass', 'Sex']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Preenche valores ausentes com a média
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Preenche valores ausentes com o valor mais frequente
        ('onehot', OneHotEncoder(drop='first'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Pipeline completo com preprocessamento e modelo
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', LogisticRegression())])
    
    # Divisão dos dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinamento do modelo
    clf.fit(X_train, y_train)
    
    # Salvando o modelo treinado
    joblib.dump(clf, model_file)
    print("Modelo treinado e salvo em", model_file)

if __name__ == "__main__":
    train_model()
