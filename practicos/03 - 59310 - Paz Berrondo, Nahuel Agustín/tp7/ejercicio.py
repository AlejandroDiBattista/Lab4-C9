import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

@st.cache
def cargar_datos():
    datos = pd.read_csv('ventas.csv')
    return datos

def crear_modelo(tamaño_entrada, tamaño_oculto, tamaño_salida):
    class RedNeuronalPrediccionVentas(nn.Module):
        def __init__(self):
            super(RedNeuronalPrediccionVentas, self).__init__()
            self.capa_oculta = nn.Linear(tamaño_entrada, tamaño_oculto)
            self.activacion = nn.Tanh()
            self.salida = nn.Linear(tamaño_oculto, tamaño_salida)
        
        def forward(self, x):
            x = self.capa_oculta(x)
            x = self.activacion(x)
            x = self.salida(x)
            return x
    return RedNeuronalPrediccionVentas()

def entrenar_modelo(modelo, criterio, optimizador, tensor_x, tensor_y, epocas):
    valores_perdida = []
    progreso = st.sidebar.progress(0)
    
    for epoca in range(epocas):
        salidas = modelo(tensor_x)
        perdida = criterio(salidas, tensor_y)
        
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        
        valores_perdida.append(perdida.item())
        
        if (epoca + 1) % (epocas // 100) == 0:
            progreso.progress((epoca + 1) / epocas)
    
    st.sidebar.success("Entrenamiento exitoso")
    return valores_perdida

def graficar_perdida(valores_perdida):
    figura, eje = plt.subplots()
    eje.plot(valores_perdida, 'g-')
    eje.set_xlabel('Época')
    eje.set_ylabel('Pérdida')
    eje.set_title('Evolución de la Pérdida durante el Entrenamiento')
    st.sidebar.pyplot(figura)

def graficar_predicciones(datos, predicciones):
    figura, eje = plt.subplots()
    eje.plot(datos['dia'], datos['ventas'], 'bo', label='Datos Reales')
    eje.plot(datos['dia'], predicciones, 'r-', label='Curva de Ajuste')
    eje.set_xlabel("Día del Mes")
    eje.set_ylabel("Ventas")
    eje.legend()
    st.pyplot(figura)

datos = cargar_datos()

st.title('Estimación de Ventas Diarias')

st.sidebar.header("Parámetros de Entrenamiento")
tasa_aprendizaje = st.sidebar.number_input("Tasa de Aprendizaje", min_value=0.0, max_value=1.0, value=0.1, step=0.001, format="%.4f")
epocas = st.sidebar.number_input("Cantidad de épocas", min_value=10, max_value=10000, value=100, step=1)
neuronas_ocultas = st.sidebar.number_input("Neuronas en Capa Oculta", min_value=1, max_value=100, value=5, step=1)
boton_entrenar = st.sidebar.button("Entrenar")

escalador_x = MinMaxScaler()
escalador_y = MinMaxScaler()

x_datos = escalador_x.fit_transform(datos['dia'].values.reshape(-1, 1))
y_datos = escalador_y.fit_transform(datos['ventas'].values.reshape(-1, 1))

tensor_x = torch.tensor(x_datos, dtype=torch.float32)
tensor_y = torch.tensor(y_datos, dtype=torch.float32)

modelo = crear_modelo(1, neuronas_ocultas, 1)
criterio = nn.MSELoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=tasa_aprendizaje)

if boton_entrenar:
    valores_perdida = entrenar_modelo(modelo, criterio, optimizador, tensor_x, tensor_y, epocas)
    graficar_perdida(valores_perdida)
    
    with torch.no_grad():
        predicciones = modelo(tensor_x).detach().numpy()
    
    predicciones = escalador_y.inverse_transform(predicciones)
    st.subheader("Estimación de Ventas Diarias")
    graficar_predicciones(datos, predicciones)
