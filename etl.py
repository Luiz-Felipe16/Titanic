import pandas as pd

def etl_process():
    input_file = 'data/titanic_data.csv'
    output_file = 'data/transformed_titanic_data.csv'
    
    df = pd.read_csv(input_file)
    
    df = df.drop(columns=['Cabin'])
    
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    etl_process()
