import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("stats.csv")
plt.plot(df['gen'], df['mejor'], label='Mejor')
plt.plot(df['gen'], df['promedio'],  label='Promedio')
plt.xlabel('Generación'); plt.ylabel('Distancia')
plt.legend(); plt.title('TSP con alg. genético  (20 individuos)')
plt.show()