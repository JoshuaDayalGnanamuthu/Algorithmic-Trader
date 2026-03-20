import numpy as np

def SafeDivide(a, b) -> float:
    if b == 0 or np.isnan(b) or np.isinf(b):
        return 0.01
    return a / b