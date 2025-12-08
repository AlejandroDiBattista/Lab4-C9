import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Definir la red neuronal
class RedNeuronal(nn.Module):
    def __init__(self, n_hidden):
        super(RedNeuronal, self).__init__()
        self.capa1 = nn.Linear(1, n_hidden)
        self.capa2 = nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        x = torch.tanh(self.capa1(x))
        x = self.capa2(x)
        return x

# Configurar la página de Streamlit
st.title('Estimación de Ventas Diarias')

# Parámetros de entrada
col1, col2 = st.columns(2)
with col1:
    tasa_aprendizaje = st.slider('Tasa de aprendizaje:', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
    epochs = st.slider('Cantidad de épocas:', min_value=10, max_value=1000, value=100, step=10)
    neuronas = st.slider('Cantidad de neuronas en capa oculta:', min_value=1, max_value=100, value=5, step=1)

# Cargar y preparar datos
@st.cache_data
def cargar_datos():
    df = pd.read_csv('ventas.csv')
    # Normalizar datos
    x = df['dia'].values.reshape(-1, 1)
    y = df['ventas'].values.reshape(-1, 1)
    x_norm = (x - x.mean()) / x.std()
    y_norm = (y - y.mean()) / y.std()
    return x_norm, y_norm, x, y

x_norm, y_norm, x_original, y_original = cargar_datos()

# Convertir datos a tensores
X = torch.FloatTensor(x_norm)
y = torch.FloatTensor(y_norm)

# Crear y entrenar el modelo
if st.button('Entrenar'):
    modelo = RedNeuronal(neuronas)
    criterio = nn.MSELoss()
    optimizador = torch.optim.SGD(modelo.parameters(), lr=tasa_aprendizaje)
    
    # Listas para almacenar el progreso
    perdidas = []
    
    # Entrenamiento
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        # Forward pass
        y_pred = modelo(X)
        perdida = criterio(y_pred, y)
        
        # Backward pass y optimización
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()
        
        # Guardar progreso
        perdidas.append(perdida.item())
        
        # Actualizar barra de progreso
        progress_bar.progress((epoch + 1) / epochs)
    
    # Graficar resultados
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Gráfica de pérdida
    ax1.plot(perdidas)
    ax1.set_title('Evolución de la función de costo')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    
    # Gráfica de predicciones
    with torch.no_grad():
        y_pred = modelo(X)
        # Desnormalizar predicciones
        y_pred_original = y_pred.numpy() * y_original.std() + y_original.mean()
    
    ax2.scatter(x_original, y_original, label='Datos Reales')
    ax2.plot(x_original, y_pred_original, 'r-', label='Curva de Ajuste')
    ax2.set_title('Ventas Diarias: Reales vs Predicciones')
    ax2.set_xlabel('Día')
    ax2.set_ylabel('Ventas')
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig)

# Mostrar explicación
st.markdown("""
### Instrucciones:
1. Ajusta los parámetros de la red neuronal usando los controles deslizantes
2. Haz clic en 'Entrenar' para iniciar el entrenamiento
3. Observa la evolución del entrenamiento y los resultados en las gráficas
""")