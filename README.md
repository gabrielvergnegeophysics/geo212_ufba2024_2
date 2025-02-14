# geo212_ufba2024_2

Biblioteca com funções implementadas de Estatística Básica, Séries de Fourier, Transformada Direta e Inversa de Fourier e Convolução. Desenvolvida pelos estudantes de Geofísica Felipe Lídio Maurício e Gabriel Vergne Mota, da Universidade Federal da Bahia, como trabalho da disciplina GEO212 à pedido do professor Diego Novais no semestre 2024.2.

## Instalação

Para utilizar a biblioteca, clone este repositório e importe o módulo no seu código Python.

```bash
pip install -r requirements.txt
```

## Funcionalidades

### 1. Geração de Sinais
- `funcao_onda_quadrada(t)`: Gera uma onda quadrada.
- `funcao_dente_de_serra(t)`: Gera uma onda dente de serra.
- `funcao_cosseno(t)`: Calcula o cosseno de um vetor de tempo.
- `funcao_cosseno_ruidosa(t)`: Gera um cosseno com ruído.
- `funcao_seno(t)`: Calcula o seno de um vetor de tempo.
- `funcao_seno_ruidosa(t)`: Gera um seno com ruído.
- `funcao_composta(t, A1, A2, A3, f1, f2, f3)`: Cria um sinal composto por senoides de diferentes frequências e amplitudes.
- `t_function(T)`: Gera um vetor de tempo para um período específico.

### 2. Análise de Fourier
- `serie_fourier(func, n, T, t)`: Calcula a Série de Fourier de uma função.
- `transformada_fourier_manual(func, t, n)`: Implementação manual da Transformada de Fourier.
- `transformada_inversa_fourier_manual(freq, F_freq, t, n)`: Implementação manual da Transformada Inversa de Fourier.

### 3. Inversão Geofísica
- `compute_jacobian(con, thick, ab2)`: Calcula o Jacobiano para um modelo geofísico.
- `forward(con, thick, ab2)`: Calcula a resposta de um modelo 1D de resistividade elétrica.
- `gauss_newton_inversion(observed_data, initial_con, thick, ab2, max_iter, tol)`: Algoritmo de inversão Gauss-Newton.
- `gauss_newton_inversion_regularizado(observed_data, initial_con, thick, ab2, par_reg, max_iter_r, tol)`: Algoritmo de inversão regularizado por Tikhonov.

### 4. Estatística
- `media(dadosx)`: Calcula a média de um conjunto de dados.
- `mediana(dadosx)`: Calcula a mediana de um conjunto de dados.
- `variancia(dadosx)`: Calcula a variância de um conjunto de dados.
- `desvio_padrao(dadosx)`: Calcula o desvio padrão de um conjunto de dados.
- `coeficiente_de_variacao(dadosx)`: Calcula o coeficiente de variação.
- `covariancia(dadosx, dadosy)`: Calcula a covariância entre dois conjuntos de dados.
- `coeficiente_de_correlacao(dadosx, dadosy)`: Calcula o coeficiente de correlação.
- `quartil1(dadosx)`, `quartil2(dadosx)`, `quartil3(dadosx)`, `quartil4(dadosx)`: Calcula os quartis de um conjunto de dados.

### 5. Processamento de Sinais
- `convolve(signal, window)`: Aplica uma convolução a um sinal.

## Exemplo de Uso

```python
from geo212_ufba2024_2 import funcao_composta, transformada_fourier_manual, transformada_inversa_fourier_manual
import numpy as np
import matplotlib.pyplot as plt

A1, A2, A3 = 2, 2, 2
f1, f2, f3 = 10, 20, 30

# Parâmetros do sinal
T = 1  # Duração total (segundos)
fs = 1000  # Taxa de amostragem (Hz)
dt = 1 / fs  # Intervalo de amostragem
t = np.arange(0, T, dt)  # Vetor de tempo
n = len(t)
sinal_original = funcao_composta(t, A1, A2, A3, f1, f2, f3)


frequencias, transformada = transformada_fourier_manual(sinal_original, t, n)
n = len(frequencias)
sinal_recuperado = transformada_inversa_fourier_manual(frequencias, transformada, t, n)

# Plotando os sinais
plt.figure(figsize=(12, 8))

# Sinal original
plt.subplot(3, 1, 1)
plt.plot(t, sinal_original, label="Sinal Original")
plt.title("Sinal Original no Domínio do Tempo")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()

# Espectro de Frequência
plt.subplot(3, 1, 2)
plt.plot(frequencias, np.abs(transformada), label="Amplitude da Transformada")
plt.title("Espectro de Frequência")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(-50, 50)
plt.legend()

# Sinal recuperado
plt.subplot(3, 1, 3)
plt.plot(t, sinal_recuperado, label="Sinal Recuperado")
plt.plot(t, sinal_original, label="Sinal Original", linestyle="--")
plt.title("Sinal Recuperado (Transformada Inversa)")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()
plt.show()

# Verifica se a recuperação foi bem-sucedida
erro = np.max(np.abs(sinal_original - sinal_recuperado))
print(f"Erro máximo entre o sinal original e o recuperado: {erro:.6e}")
```

## Contribuições
Sinta-se à vontade para abrir issues e enviar pull requests para melhorias na biblioteca.

## Licença
MIT License.