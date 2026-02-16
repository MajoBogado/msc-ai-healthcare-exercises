# models/linear_regression.py
from sklearn.linear_model import LinearRegression

def train_linear_regression(features_train, target_train):
    """
    Trains a linear regression model on training data.
    """
    model = LinearRegression()
    model.fit(features_train, target_train)
    return model
