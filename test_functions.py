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