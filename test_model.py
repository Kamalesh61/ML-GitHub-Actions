import numpy as np
from sklearn.linear_model import LinearRegression

def test_ml_model():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict([[5]])

    assert np.isclose(prediction[0], 10)

    print("ML Model Test Passed!")
