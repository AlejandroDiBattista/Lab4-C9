import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

@st.cache_data
def cargar_datos():
    df = pd.read_csv('ventas.csv')
    return df

def construir_modelo(entrada, oculta, salida):
    class RedNeuronal(nn.Module):
        def __init__(self):
            super(RedNeuronal, self).__init__()
            self.capa_oculta = nn.Linear(entrada, oculta)
            self.activacion = nn.Tanh()
            self.capa_salida = nn.Linear(oculta, salida)
        
        def forward(self, x):
            x = self.capa_oculta(x)
            x = self.activacion(x)
            x = self.capa_salida(x)
            return x
    return RedNeuronal()

def entrenar_red(modelo, criterio, optimizador, datos_x, datos_y, epocas):
    historial_perdida = []
    barra_progreso = st.sidebar.progress(0)
    
    for epoca in range(epocas):
        predicciones = modelo(datos_x)
        perdida = criterio(predicciones, datos_y)
        
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        
        historial_perdida.append(perdida.item())
        
        if (epoca + 1) % (epocas // 100) == 0:
            barra_progreso.progress((epoca + 1) / epocas)
    
    st.sidebar.success("Entrenamiento completado")
    return historial_perdida

def graficar_perdida(historial):
    fig, ax = plt.subplots()
    ax.plot(historial, 'g-')
    ax.set_xlabel('Epoca')
    ax.set_ylabel('Perdida')
    ax.set_title('Progreso de la Perdida')
    st.sidebar.pyplot(fig)

def graficar_predicciones(df, pred):
    fig, ax = plt.subplots()
    ax.plot(df['dia'], df['ventas'], 'bo', label='Datos')
    ax.plot(df['dia'], pred, 'r-', label='Predicciones')
    ax.set_xlabel("Día")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

datos = cargar_datos()

st.title('Predicción de Ventas')

st.sidebar.header("Parámetros")
tasa_aprendizaje = st.sidebar.number_input("Tasa de Aprendizaje", min_value=0.0, max_value=1.0, value=0.1, step=0.001, format="%.4f")
num_epocas = st.sidebar.number_input("Épocas", min_value=10, max_value=10000, value=100, step=1)
neuronas_ocultas = st.sidebar.number_input("Neuronas Ocultas", min_value=1, max_value=100, value=5, step=1)
boton_entrenar = st.sidebar.button("Iniciar Entrenamiento")

escalador_x = MinMaxScaler()
escalador_y = MinMaxScaler()

datos_x = escalador_x.fit_transform(datos['dia'].values.reshape(-1, 1))
datos_y = escalador_y.fit_transform(datos['ventas'].values.reshape(-1, 1))

tensor_x = torch.tensor(datos_x, dtype=torch.float32)
tensor_y = torch.tensor(datos_y, dtype=torch.float32)

modelo = construir_modelo(1, neuronas_ocultas, 1)
criterio = nn.MSELoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

if boton_entrenar:
    historial_perdida = entrenar_red(modelo, criterio, optimizador, tensor_x, tensor_y, num_epocas)
    graficar_perdida(historial_perdida)
    
    with torch.no_grad():
        predicciones = modelo(tensor_x).detach().numpy()
    
    predicciones = escalador_y.inverse_transform(predicciones)
    st.subheader("Predicción de Ventas")
    graficar_predicciones(datos, predicciones)
