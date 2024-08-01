import pandas as pd
import os

def ingest_data():
    input_file = 'data/titanic.csv'
    output_file = 'data/titanic_data.csv'
    
    df = pd.read_csv(input_file)
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    ingest_data()
