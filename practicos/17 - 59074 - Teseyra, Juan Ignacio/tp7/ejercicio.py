import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

## Leer Datos
@st.cache_data
def load_data():
    data = pd.read_csv('ventas.csv')
    return data
## Crear Red Neuronal

def create_model(input_size, hidden_size, output_size):
    class SalesPredictionNN(nn.Module):
        def __init__(self):
            super(SalesPredictionNN, self).__init__()
            self.hidden = nn.Linear(input_size, hidden_size)
            self.tanh = nn.Tanh()
            self.output = nn.Linear(hidden_size, output_size)
        
        def forward(self, x):
            x = self.hidden(x)
            x = self.tanh(x)
            x = self.output(x)
            return x
    return SalesPredictionNN()

## Entrenar Red Neuronal

def train_model(model, criterion, optimizer, x_tensor, y_tensor, epochs):
    loss_values = []
    progress = st.sidebar.progress(0)
    
    for epoch in range(epochs):
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_values.append(loss.item())
        
        if (epoch + 1) % (epochs // 100) == 0:
            progress.progress((epoch + 1) / epochs)
    
    st.sidebar.success("Entrenamiento exitoso")
    return loss_values

def plot_loss(loss_values):
    fig, ax = plt.subplots()
    ax.plot(loss_values, 'g-')
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida')
    ax.set_title('Evolución de la Pérdida durante el Entrenamiento')
    st.sidebar.pyplot(fig)

## Graficar Predicciones

def plot_predictions(data, predictions):
    fig, ax = plt.subplots()
    ax.plot(data['dia'], data['ventas'], 'bo', label='Datos Reales')
    ax.plot(data['dia'], predictions, 'r-', label='Curva de Ajuste')
    ax.set_xlabel("Día del Mes")
    ax.set_ylabel("Ventas")
    ax.legend()
    st.pyplot(fig)

data = load_data()

st.title('Estimación de Ventas Diarias')

st.sidebar.header("Parámetros de Entrenamiento")
learning_rate = st.sidebar.number_input("Tasa de Aprendizaje", min_value=0.0, max_value=1.0, value=0.1, step=0.001, format="%.4f")
epochs = st.sidebar.number_input("Cantidad de épocas", min_value=10, max_value=10000, value=100, step=1)
hidden_neurons = st.sidebar.number_input("Neuronas en Capa Oculta", min_value=1, max_value=100, value=5, step=1)
train_button = st.sidebar.button("Entrenar")

## Normalizar Datos

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_data = scaler_x.fit_transform(data['dia'].values.reshape(-1, 1))
y_data = scaler_y.fit_transform(data['ventas'].values.reshape(-1, 1))

x_tensor = torch.tensor(x_data, dtype=torch.float32)
y_tensor = torch.tensor(y_data, dtype=torch.float32)

model = create_model(1, hidden_neurons, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## Guardar Modelo

if train_button:
    loss_values = train_model(model, criterion, optimizer, x_tensor, y_tensor, epochs)
    plot_loss(loss_values)
    
    with torch.no_grad():
        predictions = model(x_tensor).detach().numpy()
    
    predictions = scaler_y.inverse_transform(predictions)
    plot_predictions(data, predictions)







