import streamlit as st
import pandas as pd
import numpy as np
    import matplotlib.pyplot as plt

st.title('Graficar f()')
c1, c2, c3 = st.columns(3)

a = c1.slider('Coefienciente A', -10, 10, 0)
b = c2.slider('Coefienciente B', -10, 10, 0)
c = c3.slider('Coefienciente C', -10, 10, 0)
c1.selectbox("Elija el tipo", ("Cuadrática", "Lineal", "Cúbica"))

st.sidebar.write(f'La ecuación es: {a}x^2 + {b}x + {c}')

minx, maxx = st.sidebar.slider('Valores de X', -10, 10, (-5,5))
def f(x):
    return a*x**2 + b*x + c

x = np.linspace(minx, maxx, 100)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Gráfico F(x)')
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.legend()
ax.grid(True)

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
