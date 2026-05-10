
import numpy as np

def predict(data):
    probability = np.random.uniform(0.2, 0.8)
    prediction = 1 if probability > 0.5 else 0
    return prediction, probability
