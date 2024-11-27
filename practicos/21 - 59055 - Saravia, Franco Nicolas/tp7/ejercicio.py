import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class ModeloVentas(nn.Module):
    def __init__(self, entradas, ocultas, salidas):
        super().__init__()
        self.capa_oculta = nn.Linear(entradas, ocultas)
        self.capa_salida = nn.Linear(ocultas, salidas)
        self.activacion = nn.ReLU()

    def forward(self, datos):
        datos = self.activacion(self.capa_oculta(datos))
        datos = self.capa_salida(datos)
        return datos

@st.cache
def cargar_datos():
    return pd.read_csv("ventas.csv")

def procesar_datos(datos):
    x_original = datos["dia"].to_numpy().reshape(-1, 1)
    y_original = datos["ventas"].to_numpy().reshape(-1, 1)
    x_min, x_max = x_original.min(), x_original.max()
    y_min, y_max = y_original.min(), y_original.max()

    x_normalizado = (x_original - x_min) / (x_max - x_min)
    y_normalizado = (y_original - y_min) / (y_max - y_min)
    
    return {
        "x": x_original, "y": y_original,
        "x_norm": x_normalizado, "y_norm": y_normalizado,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max
    }

def entrenar_modelo(modelo, datos_norm, epocas, lr):
    optimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    perdida_fn = nn.MSELoss()
    historial_perdida = []

    barra = st.sidebar.progress(0)
    for epoca in range(epocas):
        predicciones = modelo(datos_norm["x_tensor"])
        perdida = perdida_fn(predicciones, datos_norm["y_tensor"])
        
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        
        historial_perdida.append(perdida.item())
        barra.progress((epoca + 1) / epocas)
    
    return historial_perdida

def exportar_modelo(modelo, archivo="modelo_ventas.pth"):
    torch.save(modelo.state_dict(), archivo)

def mostrar_grafico(x, y, x_plot, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Datos Reales", color="blue")
    ax.plot(x_plot, y_pred, label="Predicción", color="red")
    ax.set_xlabel("Día")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

st.title("Predicción de Ventas Diarias")
st.sidebar.header("Configuración de Entrenamiento")

tasa_aprendizaje = st.sidebar.slider("Tasa de aprendizaje", 0.001, 0.1, 0.01)
epocas = st.sidebar.number_input("Épocas", min_value=10, max_value=5000, value=1000)
neuronas = st.sidebar.slider("Neuronas en capa oculta", 1, 50, 10)

if st.sidebar.button("Iniciar Entrenamiento"):
    datos = cargar_datos()
    procesado = procesar_datos(datos)

    datos_tensores = {
        "x_tensor": torch.tensor(procesado["x_norm"], dtype=torch.float32),
        "y_tensor": torch.tensor(procesado["y_norm"], dtype=torch.float32)
    }
    
    red = ModeloVentas(1, neuronas, 1)
    historial = entrenar_modelo(red, datos_tensores, epocas, tasa_aprendizaje)
    st.sidebar.success("Entrenamiento Completo")

    plt.figure()
    plt.plot(historial, label="Pérdida")
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title("Curva de Pérdida")
    plt.legend()
    st.sidebar.pyplot(plt)
    
    x_pred = np.linspace(1, 31, 100).reshape(-1, 1)
    x_pred_norm = (x_pred - procesado["x_min"]) / (procesado["x_max"] - procesado["x_min"])
    x_pred_tensor = torch.tensor(x_pred_norm, dtype=torch.float32)
    with torch.no_grad():
        y_pred = red(x_pred_tensor).numpy()
    y_pred_rescaled = y_pred * (procesado["y_max"] - procesado["y_min"]) + procesado["y_min"]

    mostrar_grafico(procesado["x"], procesado["y"], x_pred, y_pred_rescaled)