import numpy as np
from backpropagation import Backpropagation

# Dados de entrada (XOR)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Saída esperada
y = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Definindo a arquitetura: 
# 2 neurônios na entrada → 4 na 1ª camada intermediaria → 3 na 2ª camada intermediaria → 1 na saída
rna = Backpropagation(layer_sizes=[2, 4, 3, 1], activation_function='tanh', learning_rate=0.2, epochs=10000)

# Treinar a rede
rna.fit(X, y)

# Fazer previsões
output = rna.predict(X)
print("\nSaída após treinamento:")
print(output)