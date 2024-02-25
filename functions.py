import numpy as np
import math
import pandas as pd

# Função para formatar o preço
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{:.10f}".format(abs(n))

# Função para obter os dados do arquivo CSV
def getStockDataVec(file_path):
    df = pd.read_csv(file_path)
    return df['Close'].tolist()

# Função sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Função para obter o estado
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # Pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])
