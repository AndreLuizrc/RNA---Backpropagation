# 🧠 Projeto — Implementação de Redes Neurais com Backpropagation

Este projeto consiste na implementação, do zero, de uma **rede neural artificial treinada através do algoritmo de Backpropagation**, com suporte para diferentes funções de ativação, número de camadas intermediarias e hiperparâmetros customizáveis.

O objetivo é demonstrar de forma prática como redes neurais funcionam internamente, aplicando o algoritmo de retropropagação para ajuste de pesos e resolução de problemas clássicos como **AND, OR e XOR**.

---

## 🚀 Funcionalidades Principais
- ✅ Suporte para múltiplas camadas intermediarias (rede profunda).
- ✅ Implementação completa do algoritmo de **Backpropagation**.
- ✅ Escolha dinâmica da **função de ativação** (`Sigmoid` e `Tanh`).
- ✅ Inicialização aleatória dos pesos e bias.
- ✅ Treinamento no modo **Batch** (atualização dos pesos após processar todas as amostras de uma época).
- ✅ Predição de saídas após o treinamento.
- ✅ Relatórios dos erros por época (console).
- ✅ Fácil expansão e modificação.

---

## 🏗️ Arquitetura da Rede Neural
- Camada de entrada
- Uma ou mais **camadas intermediarias**
- Camada de saída

---

## 📜 Como Funciona o Backpropagation?
1. **Forward Pass:**  
   - Propagação dos dados de entrada até a saída da rede, passando por todas as camadas.  
   - A saída linear de cada neurônio é processada pela função de ativação, cuja saída alimenta a camada seguinte.

2. **Backward Pass (Backpropagation):**  
   - Cálculo do erro na camada de saída (diferença entre saída esperada e saída obtida).  
   - O erro é retropropagado pelas camadas intermediarias, ajustando os pesos proporcionalmente ao erro e à derivada da função de ativação.

3. **Atualização dos Pesos e Bias:**  
   - Realizada após o processamento de todas as amostras de uma época.  
   - Leva em conta a **taxa de aprendizado**, o erro e as ativações.

---

## 🔧 Hiperparâmetros configuráveis
- Número de camadas intermediarias e neurônios por camada.
- Função de ativação (`sigmoid` ou `tanh`).
- Taxa de aprendizado (`learning_rate`).
- Número de épocas (`epochs`).

---

## 🧠 Funções de Ativação Implementadas

| Função        | Intervalo de saída | Características                                   |
|----------------|---------------------|---------------------------------------------------|
| **Sigmoid**    | (0, 1)              | Pode sofrer com vanishing gradient, não centrada. |
| **Tanh**       | (-1, 1)             | Centragem em zero, melhora aprendizagem em XOR.  |

---

## 🏁 Importância dos Componentes da Rede Neural

### ✔️ Bias
Permite que o neurônio **desloque a função de ativação**, garantindo que ela **não seja restrita a passar pela origem (0,0)**. Isso oferece maior flexibilidade ao modelo, permitindo ajustes mais precisos, independentemente dos valores de entrada.

### ✔️ Taxa de Aprendizado
Controla o **tamanho dos ajustes dos pesos durante cada iteração**.  
- Valor alto → pode fazer o modelo divergir ou ultrapassar o mínimo.  
- Valor muito baixo → treinamento extremamente lento.  
- Um valor bem ajustado permite aprendizagem estável e eficiente.

### ✔️ Função de Ativação
Introduz **não linearidade**, permitindo que a rede aprenda padrões complexos e resolva problemas não linearmente separáveis.  
Sem uma função de ativação, a rede se comportaria como uma simples regressão linear, independentemente da profundidade.

---

