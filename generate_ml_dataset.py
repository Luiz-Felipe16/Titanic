import pandas as pd

def generate_ml_dataset():
    input_file = 'data/transformed_titanic_data.csv'
    output_file = 'data/ml_titanic_data.csv'
    
    df = pd.read_csv(input_file)
    
    # Seleção de features e target
    features = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    target = df['Survived']
    
    ml_data = pd.concat([features, target], axis=1)
    ml_data.to_csv(output_file, index=False)

if __name__ == "__main__":
    generate_ml_dataset()
