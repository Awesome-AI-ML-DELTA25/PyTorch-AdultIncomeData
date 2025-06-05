import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Setting the seed for reproducibility
torch.manual_seed(7)

def load_adultCensus_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    df.reset_index(drop=True, inplace=True)

    # We will train the model to predict income, since it is a binary variable
    target_col_y = df['income']
    predicting_col_X = df.drop('income', axis=1)


    # We will now define categorical and numerical columns, since they need to be encoded accrodingly
    categorical_col = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
    numerical_col = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week', 'net.capital']

    # Categorcal columns will be One-Hot encoded while numerical will be standard scaler, we will drop the education column

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(), categorical_col)
            ('numerical', StandardScaler(), numerical_col)
        ],
        remainder='drop'
    )

    # We will now apply this preprocessor to X, that is OneHotEncode categorical columns etc.
    processed_col_X = preprocessor.fit_transform(predicting_col_X)

    # Time to split the dataset for training and testing, we will have a 4:1 ratio
    X_train, X_test, y_train, y_test = train_test_split(processed_col_X, target_col_y, test_size=0.2, random_state=7)

    # We will now be converting these values into 2D tensors (nested lists):

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # .view(-1, 1) converts the 1D tensor into 2D, -1 means as many rows there are, and 1 means the number of columns is 1.

    # We will return these values, as well as the number of columns in the training set:
    # .shape[x] return no. of rows (x=0) and no. of columns (x=1)

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, X_train_tensor.shape[1]