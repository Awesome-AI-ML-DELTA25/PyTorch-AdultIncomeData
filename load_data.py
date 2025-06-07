import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder



def load_dataset(path):
    df = pd.read_csv(path)
    df.reset_index(drop=True, inplace=True)

    Y = df['income']
    X= df.drop('income', axis=1, inplace=False)

    categorical_cols = ['workclass','education','marital.status','occupation','relationship','race','native.country']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    
    col_trans = ColumnTransformer(
        [('num', StandardScaler(), numeric_cols),
         ('cat', OneHotEncoder(sparse_output=False), categorical_cols)],
        remainder='drop'
    )
    X_processed = col_trans.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, Y, test_size=0.3, random_state=42, shuffle=True, )
    X_cross, X_test, y_cross, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, shuffle=True, stratify=y_test)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    X_cross_tensor = torch.tensor(X_cross, dtype=torch.float32)
    y_cross_tensor = torch.tensor(y_cross.values, dtype=torch.float32).view(-1, 1)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_cross_tensor, y_cross_tensor, X_train_tensor.shape[1]


if __name__ == "__main__":
    print(load_dataset('dataset/adult_cleaned.csv'))