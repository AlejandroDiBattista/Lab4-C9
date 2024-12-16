import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.title('Estimación de Ventas Diarias')
st.sidebar.header('Parámetros de la Red Neuronal')

tasa_aprendizaje = st.sidebar.slider('Tasa de aprendizaje', 0.0, 1.0, 0.1)
epocas = st.sidebar.slider('Cantidad de épocas', 10, 10000, 100)
neuronas_ocultas = st.sidebar.slider('Cantidad de neuronas en la capa oculta', 1, 100, 5)
boton_entrenar = st.sidebar.button("Entrenar")

def cargar_datos():
    datos = pd.read_csv('ventas.csv')

    ventas_min = datos['ventas'].min()
    ventas_max = datos['ventas'].max()
    datos['ventas_normalizado'] = (datos['ventas'] - ventas_min) / (ventas_max - ventas_min)

    dias_min = datos['dia'].min()
    dias_max = datos['dia'].max()
    datos['dia_normalizado'] = (datos['dia'] - dias_min) / (dias_max - dias_min)
    
    return datos, (ventas_min, ventas_max), (dias_min, dias_max)
class RedNeuronalSimple(nn.Module):
    def __init__(self, tamano_entrada=1, tamano_oculto=5, tamano_salida=1):
        super(RedNeuronalSimple, self).__init__()
        self.oculta = nn.Linear(tamano_entrada, tamano_oculto)
        self.salida = nn.Linear(tamano_oculto, tamano_salida)
        
    def forward(self, x):
        x = torch.relu(self.oculta(x))
        x = self.salida(x)
        return x

def entrenar_red(modelo, X_entrenamiento, y_entrenamiento, tasa_aprendizaje, epocas):
    criterio = nn.MSELoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=tasa_aprendizaje)
    historial_perdida = []

    progress_text = 'Entrenamiento en progreso...'
    barra_progreso = st.progress(0)
    texto_progreso = st.empty()
    
    for epoca in range(epocas):
        modelo.train()
        optimizador.zero_grad()
        
        y_pred = modelo(X_entrenamiento)
        perdida = criterio(y_pred, y_entrenamiento)
        perdida.backward()
        optimizador.step()
        historial_perdida.append(perdida.item())
        
        progreso = (epoca + 1) / epocas
        barra_progreso.progress(progreso)
        texto_progreso.text(f'{progress_text} {progreso*100:.1f}%')
    
    return modelo, historial_perdida

datos, (ventas_min, ventas_max), (dias_min, dias_max) = cargar_datos()

st.subheader("Datos Originales de Ventas")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(datos['dia'], datos['ventas'], 'o-', label='Ventas reales')
ax.set_xlabel('Día del mes')
ax.set_ylabel('Ventas')
ax.grid(True)
ax.legend()
st.pyplot(fig)

if boton_entrenar:
    X_entrenamiento = torch.tensor(datos['dia_normalizado'].values, dtype=torch.float32).view(-1, 1)
    y_entrenamiento = torch.tensor(datos['ventas_normalizado'].values, dtype=torch.float32).view(-1, 1)
    

    modelo = RedNeuronalSimple(tamano_entrada=1, tamano_oculto=neuronas_ocultas, tamano_salida=1)
    modelo, historial_perdida = entrenar_red(modelo, X_entrenamiento, y_entrenamiento, tasa_aprendizaje, epocas)
    
    st.subheader("Evolución de la Función de Costo")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(historial_perdida, label='Pérdida')
    ax.set_xlabel('Época')
    ax.set_ylabel('Error (MSE)')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    modelo.eval()
    with torch.no_grad():
        predicciones = modelo(X_entrenamiento).view(-1).numpy()
        predicciones = (predicciones * (ventas_max - ventas_min)) + ventas_min
    
    st.subheader("Predicciones de Ventas vs Ventas Reales")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(datos['dia'], datos['ventas'], 'o-', label='Ventas reales')
    ax.plot(datos['dia'], predicciones, '--', label='Predicciones')
    ax.set_xlabel('Día del mes')
    ax.set_ylabel('Ventas')
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
    
    st.success("¡Entrenamiento finalizado con éxito!")