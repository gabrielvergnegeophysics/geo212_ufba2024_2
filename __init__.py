import numpy as np
from scipy.signal import square
import matplotlib.pyplot as plt
from scipy.integrate import quad
from math import*

__all__ = ['np', 'square', 'plt', 'quad']
def funcao_onda_quadrada(t):
  """
    Entrada:
        t (array-like): Vetor de tempo (ou valor escalar) para calcular a função onda quadrada.
    
    Saída:
        funcao_onda_quadrada (array-like): Onda quadrada, com valores -1 e 1.
    """
  funcao_onda_quadrada = -square(np.array(t))
  return funcao_onda_quadrada

def funcao_dente_de_serra(t):
  """
    Entrada:
        t (array-like): Vetor de tempo (ou valor escalar) para calcular a função dente de serra.
    
    Saída:
        funcao_dente_de_serra (array-like): Função dente de serra.
    """
  T=2*np.pi
  t = np.asarray(t)
  # Calculate the remainder of t divided by T
  remainder = np.mod(t, T)
  # Assign the remainder to funcao
  funcao_dente_de_serra = remainder - T/2
  return funcao_dente_de_serra

def funcao_cosseno(t):
  """
    Entrada:
        t (array-like): Vetor de tempo (ou valor escalar) para calcular a função cosseno.
    
    Saída:
        funcao_cosseno (array-like): Valores do cosseno de t.
    """
  funcao_cosseno = np.cos(t)
  return funcao_cosseno

def funcao_cosseno_ruidosa(t):
  """
    Entrada:
        t (array-like): Vetor de tempo (ou valor escalar) para calcular o cosseno ruidoso.
    
    Saída:
        funcao_cosseno_ruidosa (array-like): Cosseno com ruído adicionado.
    """
  if np.isscalar(t):
    t = np.array([t])
  # Apply the function element-wise using vectorization
  funcao_seno_ruidosa = funcao_cosseno(t) + np.random.normal(0, 0.025, size=t.shape)
  return funcao_seno_ruidosa

def funcao_seno(t):
  """
    Entrada:
        t (array-like): Vetor de tempo (ou valor escalar) para calcular a função seno.
    
    Saída:
        funcao_seno (array-like): Valores do seno de t.
    """
  funcao_cosseno = np.sin(t)
  return funcao_cosseno

def funcao_seno_ruidosa(t):
  """
    Entrada:
        t (array-like): Vetor de tempo (ou valor escalar) para calcular o seno ruidoso.
    
    Saída:
        funcao_seno_ruidosa (array-like): Seno com ruído adicionado.
    """
  if np.isscalar(t):
    t = np.array([t])
  # Apply the function element-wise using vectorization
  funcao_seno_ruidosa = funcao_seno(t) + np.random.normal(0, 0.025, size=t.shape)
  return funcao_seno_ruidosa

def t_function(T):
  """
    Entrada:
        T (float): O período do sinal (duração total).
    
    Saída:
        t (array-like): Vetor de tempo gerado no intervalo [-T/2, T/2] com espaçamento de 0.001.
    """
  t = np.arange(-T / 2, T / 2, 0.001)
  return t

# Serie de Fourier
def serie_fourier(func,n,T,t):
  ###################
  # Entrada: func = array = função a ser utilizada (precisa estar ajustada em 0) (exemplos prontos disponíveis: funcao_onda_quadrada, funcao_dente_de_serra, funcao_seno_ruidosa, funcao_cosseno_ruidosa
  #          n (int): Número de termos a serem somados na série.
  #          T (float): Período da função.
  #          t (array-like): Vetor de tempo para calcular a série.
  ###################
  #   Saída:
  #      sum (array-like): Série de Fourier (somatório dos termos).
  ###################
  #   Exemplo
  #
  #   n = 10
  #   T = 2*np.pi
  #   t = t_function(T)
  #   y_onda_quadrada = funcao_onda_quadrada(t_function(T))
  #   sum_onda_quadrada = serie_fourier(funcao_onda_quadrada, n, T, t)
  #
  An = []
  Bn = []
  for i in range(n):
    def fc(x):
      return func(x) * np.cos(2*np.pi*i*x/T)
    an = quad(fc, -T/2, T/2)[0] * (2.0/T)
    An.append(an)

  for i in range(n):
    def fs(x):
      return func(x) * np.sin(2*np.pi*i*x/T)
    bn = quad(fs, -T/2, T/2)[0] * (2.0/T)
    Bn.append(bn)

  sum = An[0] / 2
  for i in range(n):
    termo = An[i] * np.cos(2*np.pi*i*t/T) + Bn[i] * np.sin(2*np.pi*i*t/T)
    sum += termo

  return sum

def funcao_composta(t, A1, A2, A3, f1, f2, f3):
    """
    Entrada:
        t (array-like): Vetor de tempo.
    
    Saída:
        (array-like): Função composta por senoides de diferentes frequências e amplitudes.
    """
    return A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.cos(2 * np.pi * f2 * t) + A3 * np.sin(2 * np.pi * f3 * t)
    
# Transformada de Fourier manual
def transformada_fourier_manual(func, t, n):
    """
    Entrada:
        func (array-like): Sinal a ser transformado em Fourier.
        t (array-like): Vetor de tempo do sinal.
    
    Saída:
        freq (array-like): Frequências calculadas na transformada de Fourier.
        f_transform (array-like): Transformada de Fourier do sinal.
    """
    # Exemplo
    #
    #  T = 1  # Duração total (segundos)
    #  fs = 1000  # Taxa de amostragem (Hz)
    #  dt = 1 / fs  # Intervalo de amostragem
    #  t = np.arange(0, T, dt)  # Vetor de tempo
    #  n = len(freq)
    #  f_t = np.zeros(len(t), dtype=complex)
    #
    #  sinal_original = funcao_composta(t)
    #  frequencias, transformada = transformada_fourier_manual(sinal_original, t)
    n = len(t)
    dt = t[1] - t[0]
    freq = np.linspace(-1 / (2 * dt), 1 / (2 * dt), n, endpoint=False)
    f_transform = np.zeros(n, dtype=complex)

    for k in range(n):
        soma = 0
        for j in range(n):
            soma += func[j] * np.exp(-2j * np.pi * freq[k] * t[j]) * dt
        f_transform[k] = soma

    return freq, f_transform

# Transformada inversa de Fourier manual
def transformada_inversa_fourier_manual(freq, F_freq, t, n):
    """
    Entrada:
        freq (array-like): Frequências da transformada de Fourier.
        F_freq (array-like): Coeficientes da transformada de Fourier.
        t (array-like): Vetor de tempo para reconstrução do sinal.
    
    Saída:
        f_t (array-like): Sinal reconstruído após a transformada inversa de Fourier.
    """
    # Exemplo com a transformada direta
    #
    #  T = 1  # Duração total (segundos)
    #  fs = 1000  # Taxa de amostragem (Hz)
    #  dt = 1 / fs  # Intervalo de amostragem
    #  t = np.arange(0, T, dt)  # Vetor de tempo
    #  n = len(freq)
    #  f_t = np.zeros(len(t), dtype=complex)
    #
    #  sinal_original = funcao_composta(t)
    #  frequencias, transformada = transformada_fourier_manual(sinal_original, t)
    #  sinal_recuperado = transformada_inversa_fourier_manual(frequencias, transformada, t)
    f_t = np.zeros(len(t), dtype=complex)
    for n_idx, t_n in enumerate(t):
        soma = 0
        for k in range(n):
            soma += F_freq[k] * np.exp(2j * np.pi * freq[k] * t_n)
        f_t[n_idx] = soma

    return f_t.real

# Jacobiano
def compute_jacobian(con, thick, ab2):
    """
    Entrada:
        con (array-like): Condutividades do modelo.
        thick (array-like): Espessuras das camadas.
        ab2 (array-like): Parâmetros adicionais para o cálculo.
    
    Saída:
        jacobian (array-like): Jacobiano calculado para o modelo.
    """
    eps = 1e-6  # Perturbação para cálculo de derivada numérica
    n_layers = len(con)
    jacobian = np.zeros((len(ab2), n_layers))
    for i in range(n_layers):
        # Perturbação para a condutividade
        con_perturbed = con.copy()
        con_perturbed[i] += eps
        con_sub_perturbed = con.copy()
        con_sub_perturbed[i] -= eps
        # Respostas forward com perturbação
        forward_plus = np.array([forward(con_perturbed, thick, ab) for ab in ab2])
        forward_minus = np.array([forward(con_sub_perturbed, thick, ab) for ab in ab2])
        # Derivada numérica
        jacobian[:, i] = (forward_plus - forward_minus) / (2 * eps)
    return jacobian

def forward(con, thick, ab2):
    """
    NÃO-AUTORAL
    Calculate forward VES response with half the current electrode spacing for a 1D layered earth.

    Parameters
    ----------
    con : np.array, (n,)
        Electrical conductivity of n layers (S/m), last layer n is assumed to be infinite

    thick : np.array, (n-1,)
        Thickness of n-1 layers (m), last layer n is assumed to be infinite and does not require a thickness

    ab2 : float
        Half the current (AB/2) electrode spacing (m)

    Returns
    -------
    app_con : float
        Apparent half-space electrical conductivity (S/m)

    References
    ----------
    Ekinci, Y. L., Demirci, A., 2008. A Damped Least-Squares Inversion Program
    for the Interpretation of Schlumberger Sounding Curves, Journal of Applied Sciences, 8, 4070-4078.

    Koefoed, O., 1970. A fast method for determining the layer distribution from the raised
    kernel function in geoelectrical soundings, Geophysical Prospection, 18, 564-570.

    Nyman, D. C., Landisman, M., 1977. VES Dipole-dipole filter coefficients,
    Geophysics, 42(5), 1037-1044.
    """

    # Conductivity to resistivity and number of layers
    res = 1 / con
    lays = len(res) - 1

    # Constants
    LOG = np.log(10)
    COUNTER = 1 + (2 * 13 - 2)
    UP = np.exp(0.5 * LOG / 4.438)

    # Filter integral variable
    up = ab2 * np.exp(-10 * LOG / 4.438)

    # Initialize array
    ti = np.zeros(COUNTER)

    for ii in range(COUNTER):

        # Set bottom layer equal to its resistivity
        ti1 = res[lays]

        # Recursive formula (Koefoed, 1970)
        lay = lays
        while lay > 0:
            lay -= 1
            tan_h = np.tanh(thick[lay] / up)
            ti1 = (ti1 + res[lay] * tan_h) / (1 + ti1 * tan_h / res[lay])

        # Set overlaying layer to previous
        ti[ii] = ti1

        # Update filter integral variable
        up *= UP

    # Apply point-filter weights (Nyman and Landisman, 1977)
    res_a = 105 * ti[0] - 262 * ti[2] + 416 * ti[4] - 746 * ti[6] + 1605 * ti[8] - 4390 * ti[10] + 13396 * ti[12]
    res_a += - 27841 * ti[14] + 16448 * ti[16] + 8183 * ti[18] + 2525 * ti[20] + 336 * ti[22] + 225 * ti[24]
    res_a /= 1e4

    # Resistivity to conductivity
    return 1 / res_a

def gauss_newton_inversion(observed_data, initial_con, thick, ab2, max_iter=9, tol=1e-6):
    """
    Entradas:
    - observed_data: array, dados observados (valores reais a serem comparados com os dados previstos pelo modelo).
    - initial_con: array, valores iniciais de condutividade ou parâmetros que serão ajustados.
    - thick: array, espessuras das camadas para o modelo.
    - ab2: array, valores de medições para cada camada ou ponto de amostragem.
    - max_iter: int, número máximo de iterações (opcional, padrão é 9).
    - tol: float, tolerância de convergência (opcional, padrão é 1e-6).
    
    Saídas:
    - con: array, valores finais ajustados de condutividade ou parâmetros.
    - rmse_list2: list, lista de erros quadráticos médios (RMSE) ao longo das iterações.
    - max_iter: int, número máximo de iterações realizado.
    - delta_con: array, ajuste final realizado nos parâmetros.
    - itr: int, número de iterações realizadas até a convergência ou até o limite de iterações.
    """
    con = initial_con.copy()
    rmse_list2 = []
    itr = 0
    for itr in range(max_iter):
        # Calcula a resposta do modelo
        predicted_data = np.array([forward(con, thick, ab) for ab in ab2])
        # Calcula o resíduo
        residual = observed_data - predicted_data

        rmse2 = np.sqrt(np.mean(residual**2))
        rmse_list2.append(rmse2)

        itr += 1

        if rmse2 < tol:
          print("Convergência alcançada na iteração", itr)
          break

        # Calcula o jacobiano
        J = compute_jacobian(con, thick, ab2)

        # Decomposição QR do Jacobiano
        Q, R = np.linalg.qr(J)

        # Resolve o sistema de equações normais usando decomposição QR para decompor o Jacobiano em duas matrizes
        delta_con = np.linalg.solve(R, np.dot(Q.T, residual))

        # Atualiza os parâmetros
        con += delta_con

        if np.linalg.norm(delta_con) < tol:
            print(f"Convergência atingida em {itr + 1} iterações.")
            break
    else:
        print("Número máximo de iterações atingido sem convergência.")
    return con, rmse_list2, max_iter, delta_con, itr

#Regularização Tikhonov
def gauss_newton_inversion_regularizado(observed_data, initial_con, thick, ab2, par_reg, max_iter_r=9, tol=1e-6):
    """
    Entradas:
    - observed_data: array, dados observados (valores reais a serem comparados com os dados previstos pelo modelo).
    - initial_con: array, valores iniciais de condutividade ou parâmetros que serão ajustados.
    - thick: array, espessuras das camadas para o modelo.
    - ab2: array, valores de medições para cada camada ou ponto de amostragem.
    - par_reg: float, parâmetro de regularização de Tikhonov.
    - max_iter_r: int, número máximo de iterações (opcional, padrão é 9).
    - tol: float, tolerância de convergência (opcional, padrão é 1e-6).
    
    Saídas:
    - con: array, valores finais ajustados de condutividade ou parâmetros.
    - rmse_list: list, lista de erros quadráticos médios (RMSE) ao longo das iterações.
    - max_iter_r: int, número máximo de iterações realizado.
    - delta_con: array, ajuste final realizado nos parâmetros.
    - itr: int, número de iterações realizadas até a convergência ou até o limite de iterações.
    """
    con = initial_con.copy()
    rmse_list = []
    n_layers = len(con)
    # Matriz de regularização de Tikhonov (diferenças de primeira ordem)
    L = np.eye(n_layers) - np.diag(np.ones(n_layers - 1), k=1) - np.diag(np.ones(n_layers - 1), k=-1)
    # Subtrai, da matriz identidade, a matriz identidade deslocada uma posição acima.

    # A subtração dessas duas matrizes resulta em uma matriz que calcula a diferença entre um parâmetro e o seu vizinho imediato (diferença finita de primeira ordem).
    for itr in range(max_iter_r):
        # Calcula a resposta do modelo
        predicted_data = np.array([forward(con, thick, ab) for ab in ab2])
        # Calcula o resíduo
        residual = observed_data - predicted_data

        rmse = np.sqrt(np.mean(residual**2))
        rmse_list.append(rmse)

        if rmse < tol:
          break

        # Calcula o jacobiano
        J = compute_jacobian(con, thick, ab2)
        # Termo de regularização
        JTJ = J.T @ J
        LTL = L.T @ L
        JT_residual = J.T @ residual
        # Resolve o sistema linear regularizado
        delta_con = np.linalg.solve(JTJ + par_reg * LTL, JT_residual)
        # Atualiza os parâmetros
        con += delta_con
        # Verifica o critério de convergência
        if np.linalg.norm(delta_con) < tol:
            print(f"Convergência atingida em {itr + 1} iterações.")
            break
    else:
        print("Número máximo de iterações atingido sem convergência.")

    return con, rmse_list, max_iter_r, delta_con, itr

# Convolução
def convolve(signal, window):
    """
    Entradas:
    - signal: array, sinal de entrada (normalmente, um vetor de amostras de uma série temporal).
    - window: array, janela (normalmente, um filtro ou kernel utilizado na convolução).

    Saídas:
    - output: array, sinal resultante da convolução do sinal com a janela.
    """
    ###
    # Exemplo
    #
    # 
    # x = np.linspace(0, 10, 300)
    # sig = np.sin(2 * np.pi * x)
    # win = np.kaiser(50, 1)
    # filtered = convolve(sig, win) / sum(win)
    ###
    signal_len = len(signal)
    window_len = len(window)
    output_len = signal_len

    output = np.zeros(output_len)

    for i in range(output_len):
        for j in range(window_len):
            if i - j + (window_len // 2) >= 0 and i - j + (window_len // 2) < signal_len:
                output[i] += signal[i - j + (window_len // 2)] * window[j]

    return output

### EXTRA: Funções de estatística
def media(dadosx):
  """
  Esta função calcula a média aritmética dos dados da lista ou array "dadosx".

  Entrada:
    dadosx - lista ou array de entrada
  Saída:
    resultado_media - média aritmética da lista ou array "dadosx".
  """

  if int(len(dadosx)) == 0:
    return "ERRO: a lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "A média é zero, pois a lista ou array de entrada não possui nenhum elemento não-nulo."
  else:
    total = 0
    for i in range (len(dadosx)):
      total = total + dadosx[i]
    resultado_media = total/len(dadosx)
    return resultado_media

def mediana(dadosx):
  """
  Esta função calcula a mediana dos dados da lista ou array "dadosx".

  Entrada:
    dadosx - lista ou array de entrada
  Saída:
    med - mediana da lista ou array "dadosx".
  """

  if int(len(dadosx)) == 0:
    return "ERRO: a lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "A mediana é zero, pois a lista ou array de entrada não possui nenhum elemento não-nulo."
  else:
    metade_do_tamanho = int((len(dadosx))/2)
    resto = (len(dadosx))%2
    if resto != 0:
      return dadosx[metade_do_tamanho]
    if resto == 0:
      m = metade_do_tamanho
      n = m-1
      med = ( dadosx[m] + dadosx[n] ) / 2
      return med

def variancia(dadosx):
  """
  Esta função calcula a variância dos dados da lista ou array "dadosx".

  Entrada:
    dadosx - lista ou array de entrada
  Saída:
    var - mediana da lista ou array "dadosx".
  """

  if int(len(dadosx)) == 0:
    return "ERRO: a lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "A variância é zero, pois a lista ou array de entrada não possui nenhum elemento não-nulo."

  total = 0
  for i in range (len(dadosx)):
    dadosx[i] = float(dadosx[i])
    total += ((dadosx[i]) - media(dadosx) ) ** 2
  var = (total / len(dadosx))
  return var

def desvio_padrao(dadosx):
  """
  Esta função calcula a medida de desvio padrão dos dados da lista ou array "dadosx".

  Entrada:
    dadosx - lista ou array de entrada
  Saída:
    desv_padrao - desvio padrão da lista ou array "dadosx".
  """
  if int(len(dadosx)) == 0:
    return "ERRO: a lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "ERRO: a lista ou array de entrada não possui nenhum elemento não-nulo."
  desv_padrao = variancia(dadosx) ** (1/2)
  return desv_padrao

def coeficiente_de_variacao(dadosx):
  """
  Esta função calcula a medida do coeficiente de variação entre os dados da lista ou array "dadosx".

  Entrada:
    dadosx - lista ou array de entrada
    dadosy - lista ou array de entrada
  Saída:
    coef_var - coeficiente de variação da lista ou array "dadosx".
  """
  if int(len(dadosx)) == 0:
    return "ERRO: a lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "ERRO: a lista ou array de entrada não possui nenhum elemento não-nulo e a função coeficiente de variação não pode ser calculada, pois não existe divisão por zero."
  coef_var = desvio_padrao(dadosx) / media(dadosx)
  return coef_var

def covariancia(dadosx, dadosy):
  """
  Esta função calcula a covariância entre duas listas ou entre dois arrays.

  Entrada:
    dadosx - lista ou array de entrada
    dadosy - lista ou array de entrada
  Saída:
    cov - covariância entre os conjuntos de dados "dadosx" e "dadosy".
  """
  if int(len(dadosx)) == 0:
    return "ERRO: a primeira lista ou array de entrada está vazia, não possui elementos."
  if int(len(dadosy)) == 0:
    return "ERRO: a segunda lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "A covariância é zero, pois a primeira lista ou array de entrada não possui nenhum elemento não-nulo."
  if all(y == 0 for y in dadosy):
    return "A covariância é zero, pois a segunda lista ou array de entrada não possui nenhum elemento não-nulo."
  if len(dadosx) != len(dadosy):
    return "ERRO: os dois conjuntos de dados devem ter o mesmo tamanho para que se possa calcular a covariância."
  else:
    x_atualizado = []
    y_atualizado = []
    produto = []
    m = 0
    n = 0
    p = 0
    for i in (dadosx):
      m = (i - media(dadosx))
      x_atualizado.append(m)
      m = 0
    for e in (dadosy):
      n = (e - media(dadosy))
      y_atualizado.append(n)
      n = 0
    for i in range (len(y_atualizado)):
      produto.append(x_atualizado[i]*y_atualizado[i])
      somatorio = sum(produto)
      p = p+1
    cov = somatorio / (p-1)
    return cov

def coeficiente_de_correlacao(dadosx, dadosy):
  """
  Esta função calcula o coeficiente de correlação entre duas listas ou entre dois arrays.

  Entrada:
    dadosx - lista ou array de entrada
    dadosy - lista ou array de entrada
  Saída:
    rxy - coeficiente de correlação entre os conjuntos de dados "dadosx" e "dadosy".
  """

  if (len(dadosx)) == 0:
    return "ERRO: a primeira lista ou array de entrada está vazia, não possui elementos."
  if (len(dadosy)) == 0:
    return "ERRO: a segunda lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "ERRO: a primeira lista ou array de entrada possui elemento não-nulo e, como não existe divisão por zero, é impossível executar o comando."
  if all(y == 0 for y in dadosy):
    return "ERRO: a segunda lista ou array de entrada possui elemento não-nulo e, como não existe divisão por zero, é impossível executar o comando."
  if len(dadosx) != len(dadosy):
    return "ERRO: os dois conjuntos de dado devem ter o mesmo tamanho para que se possa calcular a covariância."

  #x_=media(dadosx)
  #y_=media(dadosy)

  soma_n=0
  soma_d1=0
  soma_d2=0
  for i in range(len(dadosx)):
    soma_n+=(dadosx[i]-media(dadosx))*(dadosy[i]-media(dadosy))
    soma_d1+=((dadosx[i]-media(dadosx))**2)
    soma_d2+=((dadosy[i]-media(dadosy))**2)
  rxy=soma_n/((soma_d1*soma_d2)**(1/2))
  return rxy

def quartil1(dadosx):
  """
  Esta função calcula o primeiro quartil de uma lista ou array. O primeiro quartil é o último elemento do conjunto
  que vai até 25% do conjunto de dados ordenado de maneira crescente. Para casos onde o conjunto de dados não é
  divisível por quatro sem resto, então é realizado um cálculo de soma do termo da posição resultante da divisão
  inteira, com a diferença entre o termo da posição seguinte e o termo da posição resultante da divisão inteira,
  multiplicado pela parte decimal da divisão do tamanho do conjunto de dados por quatro. Também chamado de percentil
  25, o primeiro quartil é uma medida bastante usada na Estatística Descritiva.

  Entrada:
    dadosx - lista ou array de entrada
  Saída:
    q1 - primeiro da entrada ordenada crescentemente.
  """
  if (len(dadosx)) == 0:
    return "ERRO: a lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "O primeiro quartil é zero, pois a lista ou array de entrada não possui nenhum elemento não-nulo."
  x = sorted(dadosx)
  m = len(dadosx)
  n = (m//4) - 1
  if m%4 == 0:
    q1 = dadosx[n]
  else:
    l = (n%4)/2
    q1 = (x[n] + l*(x[n+1]-x[n]))
  return q1

def quartil2(dadosx):
  """
  Esta função calcula o segundo quartil de uma lista ou array. O primeiro quartil é o último elemento do conjunto
  que vai até 50% do conjunto de dados ordenado de maneira crescente. Para casos onde o conjunto de dados não é
  divisível por quatro sem resto, então é realizado um cálculo de soma do termo da posição resultante da divisão
  inteira, com a diferença entre o termo da posição seguinte e o termo da posição resultante da divisão inteira,
  multiplicado pela parte decimal da divisão do tamanho do conjunto de dados por quatro. Também chamado de mediana
  ou percentil 50, o segundo quartil é uma medida bastante usada na Estatística Descritiva.

  Entrada:
    dadosx - lista ou array de entrada
  Saída:
    mediana(dadosx) - segundo quartil ou mediana da entrada ordenada crescentemente.
  """
  if (len(dadosx)) == 0:
    return "ERRO: a lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "O primeiro quartil é zero, pois a lista ou array de entrada não possui nenhum elemento não-nulo."
  return(mediana(dadosx))

def quartil3(dadosx):
  """
  Esta função calcula o terceiro quartil de uma lista ou array. O terceiro quartil é o último elemento do conjunto
  que vai até 75% do conjunto de dados ordenado de maneira crescente. Para casos onde o conjunto de dados não é
  divisível por quatro sem resto, então é realizado um cálculo de soma do termo da posição resultante da divisão
  inteira, com a diferença entre o termo da posição seguinte e o termo da posição resultante da divisão inteira,
  multiplicado pela parte decimal da divisão do tamanho do conjunto de dados por quatro. Também chamado de percentil
  75, o segundo quartil é uma medida bastante usada na Estatística Descritiva.

  Entrada:
    dadosx - lista ou array de entrada
  Saída:
    q3 - terceiro quartil ou mediana da entrada ordenada crescentemente.
  """
  if (len(dadosx)) == 0:
    return "ERRO: a lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "O primeiro quartil é zero, pois a lista ou array de entrada não possui nenhum elemento não-nulo."
  x = sorted(dadosx)
  m = len(dadosx)
  o = 3*(m/4)
  n = int(3*(m/4))
  p = o - n
  if m%4 == 0:
    q3 = dadosx[n]
  else:
    q3 = (x[n-1] + p*(x[n]-x[n-1]))
  return q3

def quartil4(dadosx):
  """
  Esta função calcula o quarto quartil de uma lista ou array. O quarto quartil é o último elemento do conjunto,
  ordenado de maneira crescente. Nesse sentido, é o maior elemento positivo da lista ou array de entradas. Assim,
  o quarto quartil é uma medida bastante usada na Estatística Descritiva.

  Entrada:
    dadosx - lista ou array de entrada
  Saída:
    mediana(dadosx) - segundo quartil ou mediana da entrada ordenada crescentemente.
  """
  if (len(dadosx)) == 0:
    return "ERRO: a lista ou array de entrada está vazia, não possui elementos."
  if all(x == 0 for x in dadosx):
    return "O primeiro quartil é zero, pois a lista ou array de entrada não possui nenhum elemento não-nulo."
  x = sorted(dadosx)
  m = (len(dadosx))
  q4 = x[m-1]
  return q4
