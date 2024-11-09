import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(layout="wide")

data = pd.read_csv('ventas.csv')

scaler = MinMaxScaler()
data['ventas_norm'] = scaler.fit_transform(data[['ventas']])
X = data['dia'].values.reshape(-1, 1)
y = data['ventas_norm'].values.reshape(-1, 1)

st.sidebar.header("Parámetros de Entrenamiento")

col1, col2 = st.sidebar.columns(2)
learning_rate = col1.number_input("Aprendizaje", value=0.01, step=0.001, format="%.4f")
epochs = col2.number_input("Repeticiones", min_value=10, max_value=10000, value=1000, step=100)

hidden_neurons = st.sidebar.number_input("Neuronas Capa Oculta", min_value=1, max_value=100, value=10)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.hidden(x)
        out = self.relu(out)
        out = self.output(out)
        return out

if st.sidebar.button("Entrenar"):
    model = NeuralNet(1, hidden_neurons, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    inputs = torch.tensor(X, dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.float32)

    losses = []
    for epoch in range(epochs):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    st.sidebar.write(f"Epoch {epochs}/{epochs} - Error: {loss.item():.6f}")

    st.sidebar.success("Entrenamiento exitoso")

    st.sidebar.write("### Pérdidas")
    plt.figure(figsize=(3, 2))
    plt.plot(losses, label="Pérdidas", color="green")
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.legend()
    st.sidebar.pyplot(plt)

    st.write("## Estimación de Ventas Diarias")
    container = st.empty()
    with container:
        plt.figure(figsize=(7, 5))
        with torch.no_grad():
            predicted = model(inputs).detach().numpy()
        plt.scatter(X, y, color="blue", label="Datos Reales")
        plt.plot(X, predicted, color="red", label="Curva de Ajuste")
        plt.xlabel("Día del Mes")
        plt.ylabel("Ventas")
        plt.legend()
        st.pyplot(plt, use_container_width=True)
