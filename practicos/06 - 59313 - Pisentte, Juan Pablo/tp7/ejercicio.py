import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

@st.cache
def cargar_datos():
    ventas_data = pd.read_csv('ventas.csv')
    return ventas_data

def construir_modelo(entrada_dim, capa_oculta_dim, salida_dim):
    class RedNeuronalPrediccionVentas(nn.Module):
        def __init__(self):
            super(RedNeuronalPrediccionVentas, self).__init__()
            self.capa_oculta = nn.Linear(entrada_dim, capa_oculta_dim)
            self.activacion = nn.Tanh()
            self.capa_salida = nn.Linear(capa_oculta_dim, salida_dim)
        
        def forward(self, x):
            x = self.capa_oculta(x)
            x = self.activacion(x)
            x = self.capa_salida(x)
            return x
    return RedNeuronalPrediccionVentas()

def entrenar_modelo(modelo, funcion_error, optimizador, entradas_tensor, salidas_tensor, epocas):
    valores_perdida = []
    progreso = st.sidebar.progress(0)
    
    for epoca in range(epocas):
        predicciones = modelo(entradas_tensor)
        perdida = funcion_error(predicciones, salidas_tensor)
        
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        
        valores_perdida.append(perdida.item())
        
        if (epoca + 1) % (epocas // 100) == 0:
            progreso.progress((epoca + 1) / epocas)
    
    st.sidebar.success("Entrenamiento completado")
    return valores_perdida

def graficar_perdida(valores_perdida):
    fig, ax = plt.subplots()
    ax.plot(valores_perdida, 'g-')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida')
    ax.set_title('Evolución de la Pérdida durante el Entrenamiento')
    st.sidebar.pyplot(fig)

def graficar_predicciones(datos, predicciones):
    fig, ax = plt.subplots()
    ax.plot(datos['dia'], datos['ventas'], 'bo', label='Datos Reales')
    ax.plot(datos['dia'], predicciones, 'r-', label='Curva de Ajuste')
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

# Cargar datos
datos = cargar_datos()

st.title('Estimación de Ventas Diarias')

# Parámetros en la barra lateral
st.sidebar.header("Parámetros de Entrenamiento")
tasa_aprendizaje = st.sidebar.number_input("Tasa de Aprendizaje", min_value=0.0, max_value=1.0, value=0.1, step=0.001, format="%.4f")
epocas = st.sidebar.number_input("Cantidad de épocas", min_value=10, max_value=10000, value=100, step=1)
neuronas_ocultas = st.sidebar.number_input("Neuronas en Capa Oculta", min_value=1, max_value=100, value=5, step=1)
boton_entrenar = st.sidebar.button("Entrenar")

# Escalado de datos
escalador_x = MinMaxScaler()
escalador_y = MinMaxScaler()

x_data = escalador_x.fit_transform(datos['dia'].values.reshape(-1, 1))
y_data = escalador_y.fit_transform(datos['ventas'].values.reshape(-1, 1))

entradas_tensor = torch.tensor(x_data, dtype=torch.float32)
salidas_tensor = torch.tensor(y_data, dtype=torch.float32)

# Crear modelo
modelo = construir_modelo(1, neuronas_ocultas, 1)
funcion_error = nn.MSELoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

if boton_entrenar:
    valores_perdida = entrenar_modelo(modelo, funcion_error, optimizador, entradas_tensor, salidas_tensor, epocas)
    graficar_perdida(valores_perdida)
    
    with torch.no_grad():
        predicciones = modelo(entradas_tensor).detach().numpy()
    
    predicciones = escalador_y.inverse_transform(predicciones)
    st.subheader("Estimación de Ventas Diarias")
    graficar_predicciones(datos, predicciones)
