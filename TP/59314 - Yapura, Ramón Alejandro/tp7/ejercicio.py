import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('ventas.csv')

class PrediccionVentas(nn.Module):
    def _init_(self, neuronas_intermedias):
        super(PrediccionVentas, self)._init_()
        self.capa_entrada = nn.Linear(1, neuronas_intermedias)
        self.capa_salida = nn.Linear(neuronas_intermedias, 1)
        
    def forward(self, entrada):
        capa_intermedia = torch.relu(self.capa_entrada(entrada))
        salida = self.capa_salida(capa_intermedia)
        return salida

def ejecutar_entrenamiento(tasa_aprendizaje, ciclos_entrenamiento, neuronas_intermedias):
    predictor = PrediccionVentas(neuronas_intermedias)
    optimizador = torch.optim.Adam(predictor.parameters(), lr=tasa_aprendizaje)
    funcion_perdida = nn.MSELoss()
    historial_perdidas = []
    
    with st.container():
        barra_progreso = st.progress(0, text='Iniciando entrenamiento...')
        
        for ciclo in range(ciclos_entrenamiento):
            # Preparar datos
            datos_entrada = torch.tensor(df['dia'].values.reshape(-1, 1), dtype=torch.float32)
            datos_objetivo = torch.tensor(df['ventas'].values.reshape(-1, 1), dtype=torch.float32)
            
            # Proceso de entrenamiento
            predicciones = predictor(datos_entrada)
            perdida = funcion_perdida(predicciones, datos_objetivo)
            
            # Actualización de pesos
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
            
            # Registro y actualización de progreso
            historial_perdidas.append(perdida.item())
            barra_progreso.progress((ciclo + 1) / ciclos_entrenamiento, 
                                 text=f'Ciclo {ciclo + 1}/{ciclos_entrenamiento} - Error: {perdida.item():.6f}')
    
    return predictor, historial_perdidas

# Interfaz de usuario
st.title('Predicción de Ventas Diarias')

# Panel lateral de configuración
st.sidebar.header('Configuración del Modelo')

# Controles divididos en columnas
col_izq, col_der = st.sidebar.columns(2)

with col_izq:
    tasa = st.number_input('Tasa de Aprendizaje', min_value=0.0, max_value=1.0, value=0.1, step=0.01)
with col_der:
    ciclos = st.number_input('Ciclos', min_value=10, max_value=10000, value=100, step=10)

neuronas = st.sidebar.number_input('Neuronas Intermedias', min_value=1, max_value=100, value=5, step=1)

if st.sidebar.button('Iniciar Entrenamiento'):
    with st.sidebar.container():
        modelo, historico_perdidas = ejecutar_entrenamiento(tasa, ciclos, neuronas)
        st.success('Entrenamiento completado')
    
    # Visualización de resultados
    figura1, grafica1 = plt.subplots(figsize=(8, 6))
    grafica1.scatter(df['dia'], df['ventas'], color='blue', label='Datos Originales')
    grafica1.plot(df['dia'], 
                 modelo(torch.tensor(df['dia'].values.reshape(-1, 1), dtype=torch.float32)).detach().numpy(), 
                 color='red', 
                 label='Predicción')
    grafica1.set_xlabel('Día')
    grafica1.set_ylabel('Ventas')
    grafica1.set_title('Predicción de Ventas Diarias')
    grafica1.legend()
    st.pyplot(figura1)
    
    figura2, grafica2 = plt.subplots(figsize=(8, 6))
    grafica2.plot(historico_perdidas, color='green')
    grafica2.set_xlabel('Ciclo')
    grafica2.set_ylabel('Error')
    grafica2.set_title('Evolución del Error')
    st.sidebar.pyplot(figura2)