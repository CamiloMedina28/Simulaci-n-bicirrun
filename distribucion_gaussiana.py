import numpy as np
import matplotlib.pyplot as plt
from functools import partial

def demanda_gaussiana(t, mus, sigmas, amplitudes):
    """
    t: hora en formato decimal (ej: 7.5 para 7:30)
    mus: lista de horas pico
    sigmas: lista de anchos de pico
    amplitudes: lista de pesos (intensidad por pico)
    """
    t = np.array(t)
    total = 0
    for μ, σ, A in zip(mus, sigmas, amplitudes):
        total += A * np.exp(-((t - μ)**2) / (2 * σ**2))
    return total


mus = [6.75, 9, 11, 13, 14, 16]

sigmas = [0.20, 0.30, 0.30, 0.25, 0.25, 0.30]

amplitudes = [1.0, 0.8, 0.6, 0.7, 0.5, 0.9]

# Calcular demanda a las 10:30 am (10.5)
d = demanda_gaussiana(10.5, mus, sigmas, amplitudes)
print(d)

demanda_diaria_fijada = partial(
    demanda_gaussiana,
    mus=mus,
    sigmas=sigmas,
    amplitudes=amplitudes
)

x = np.linspace(6.30, 16.30, 100)
y = demanda_diaria_fijada(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Demanda del día', color='purple')

plt.title('Modelo de Demanda Gaussiana Diaria')
plt.xlabel('Hora del día (h)')
plt.ylabel('Nivel de Demanda')
# plt.xticks(np.arange(0, 25, 4)) # Marcas cada 4 horas
plt.grid(True)
plt.legend()
plt.show()