import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder



def load_adult_income_data(path="dataset/adult_income_clean.csv"):
    df = pd.read_csv(path)
    
    # Binary encode the target column: 1 if '>50k' else 0
    df['income'] = df['income'].apply(lambda x: 1 if '>50k' in x.lower() else 0)
    
    X = df.drop('income', axis=1)
    y = df['income']

    categorical_cols = [
        'workclass', 'education', 'marital.status', 'occupation', 
        'relationship', 'race', 'sex', 'native.country'
    ]
    numeric_cols = [
        'age', 'fnlwgt', 'education.num', 'capital.gain', 
        'capital.loss', 'hours.per.week', 'net.capital'
    ]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    X_processed = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_tensor = torch.tensor(
        X_train.toarray() if hasattr(X_train, "toarray") else X_train,
        dtype=torch.float32
    )
    X_test_tensor = torch.tensor(
        X_test.toarray() if hasattr(X_test, "toarray") else X_test,
        dtype=torch.float32
    )
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    input_dim = X_train_tensor.shape[1]
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_dim

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, input_dim = load_adult_income_data("dataset/adult_income_clean.csv")
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}, Features: {input_dim}")
