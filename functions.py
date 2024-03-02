import numpy as np
import math
import pandas as pd

# Função para formatar o preço
def formatPriceList(price_list):
    formatted_prices = []
    for price in price_list:
        formatted_prices.append(("-$" if price < 0 else "$") + "{:.10f}".format(abs(price)))
    return formatted_prices

def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{:.10f}".format(abs(n))

# Função para obter os dados do arquivo CSV
# Função para obter os dados OHLC do arquivo CSV
def getStockDataVec(file_path):
    df = pd.read_csv(file_path)
    return df[['Open', 'High', 'Low', 'Close']].values.tolist()

# Função sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Função para obter o estado
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # Pad with t0
    res = []
    for i in range(n - 1):
        # Ajuste os pesos conforme necessário
        # Aqui, atribuímos 60% ao preço de fechamento (Close) e 10% a cada uma das outras componentes
        weight_close = 0.6
        weight_open = 0.1
        weight_high = 0.1
        weight_low = 0.1
        weighted_close = weight_close * sigmoid(block[i + 1][3] - block[i][3])
        weighted_open = weight_open * sigmoid(block[i + 1][0] - block[i][0])
        weighted_high = weight_high * sigmoid(block[i + 1][1] - block[i][1])
        weighted_low = weight_low * sigmoid(block[i + 1][2] - block[i][2])
        state_value = weighted_close + weighted_open + weighted_high + weighted_low
        res.append(state_value)  
    return np.array([res])

