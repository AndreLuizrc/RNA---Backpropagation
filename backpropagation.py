import numpy as np

class Backpropagation:
    def __init__(self, layer_sizes, activation_function='sigmoid', learning_rate=0.1, epochs=10000):
        """
        layer_sizes: lista com o número de neurônios por camada.
            Ex: [2, 4, 3, 1] → 2 na entrada, 4 e 3 nas camadas ocultas, 1 na saída.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_layers = len(layer_sizes)
        
        if activation_function == 'sigmoid':
            self.activation = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation_function == 'tanh':
            self.activation = self.tanh
            self.activation_derivative = self.tanh_derivative
        else:
            raise ValueError(f'Função de ativação "{activation_function}" não implementada.')
        
        print(f"Função escolhida: {activation_function}")

        np.random.seed(42)  # Para resultados reprodutíveis

        # Inicializando pesos e bias para cada camada
        self.weights = [] 
        self.biases = []

        for i in range(self.n_layers - 1): # loop para incializar os pesos para cada camada com valores aleátorios 
            weight_matrix = np.random.rand(layer_sizes[i], layer_sizes[i + 1])  
            bias_vector = np.random.rand(1, layer_sizes[i + 1])
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    # Função de ativação sigmoid
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivada da sigmoid
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    # Função de ativação tangente hiperbolica
    def tanh(self, x):
        return np.tanh(x)

    # derivada da Tanh
    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    # Forward pass
    def forward(self, X):
        activations = [X] # Inicializa activations com as entradas inciais
        inputs = []

        for w, b in zip(self.weights, self.biases): # loop para realizar a propagação entre todas as camadas
            linear_output = np.dot(activations[-1], w) + b # realiza calculo da saida linear para todos os pesos e entradas da camada
            inputs.append(linear_output) # Armazena a saída linear
            a = self.activation(linear_output) # chama a função de ativação para todas as saidas lineares geradas da camada atual
            activations.append(a)

        return activations, inputs

    # Backward pass (Backpropagation)
    def backward(self, X, y, activations, inputs):
        deltas = [None] * (self.n_layers - 1)  # Erros locais, vetor inicialmente preenchido com None

        # Erro na camada de saída
        error = y - activations[-1] # Calcula o erro da camada de saída: y_real - y_predito, para todas as amostras 
        deltas[-1] = error * self.activation_derivative(activations[-1]) # Guarda o erro da camada de saída

        # Backpropagation para camadas ocultas
        for l in reversed(range(len(deltas) - 1)): # O loop é realizado na ordem reversa para propagar o erro da camada mais próxima a saída até a entrada
            delta_next = deltas[l + 1]
            w_next = self.weights[l + 1]
            delta = np.dot(delta_next, w_next.T) * self.activation_derivative(activations[l + 1]) # Erro da camada atual: Somatorio(Ei+1 . wi) * f(a)
            deltas[l] = delta

        # Atualização dos pesos e bias
        for l in range(len(self.weights)): # loop para atualizar o peso de todas as camadas
            self.weights[l] += self.learning_rate * np.dot(activations[l].T, deltas[l]) # Atualiza os pesos da camada atual 
            self.biases[l] += self.learning_rate * np.sum(deltas[l], axis=0, keepdims=True) # Atualiza o peso do bias atual

    # Treinamento da rede
    def fit(self, X, y):
        for epoch in range(self.epochs):
            activations, inputs = self.forward(X)
            self.backward(X, y, activations, inputs)

            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - activations[-1]))
                print(f'Época {epoch}, Erro: {loss:.6f}')

    # Predição
    def predict(self, X):
        activations, _ = self.forward(X)
        return activations[-1]
