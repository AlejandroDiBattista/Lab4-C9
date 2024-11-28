import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Estimación de Ventas Diarias", layout="wide")
st.title('Estimación de Ventas Diarias')

class VentasNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VentasNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x

@st.cache_data
def load_data():
    data = pd.read_csv("ventas_diarias.csv") 
    return data

data = load_data()
st.write("Datos cargados:", data.head())

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['fecha', 'ventas']])

X = torch.FloatTensor(scaled_data[:, 0].reshape(-1, 1))
y = torch.FloatTensor(scaled_data[:, 1].reshape(-1, 1))

model = VentasNet(input_size=1, hidden_size=10, output_size=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = st.slider("Número de épocas", min_value=100, max_value=10000, value=1000, step=100)
losses = []

if st.button("Entrenar modelo"):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            st.write(f'Época [{epoch+1}/{epochs}], Pérdida: {loss.item():.4f}')

    torch.save(model.state_dict(), 'modelo_ventas.pth')
    st.success("Modelo entrenado y guardado con éxito!")

if st.button("Cargar modelo guardado"):
    model.load_state_dict(torch.load('modelo_ventas.pth'))
    st.success("Modelo cargado con éxito!")

if st.button("Mostrar predicciones"):
    model.eval()
    with torch.no_grad():
        predicted = model(X).numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data['fecha'], data['ventas'], color='blue', label='Datos reales')
    ax.plot(data['fecha'], scaler.inverse_transform(predicted), color='red', label='Predicciones')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Ventas')
    ax.legend()
    st.pyplot(fig)

new_date = st.number_input("Ingrese una nueva fecha para predecir (en formato numérico)", min_value=0.0, max_value=1.0, value=0.5)
if st.button("Predecir"):
    model.eval()
    with torch.no_grad():
        new_input = torch.FloatTensor([[new_date]])
        prediction = model(new_input)
        st.write(f"Predicción de ventas para la fecha {new_date}: {scaler.inverse_transform(prediction.numpy())[0][0]:.2f}")