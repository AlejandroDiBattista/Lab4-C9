import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.title('estimacion de ventas diarias')

st.header('cargar y mostrar datos')
data_file = 'ventas.csv'
data = pd.read_csv(data_file)
st.write('datos cargados')
st.write(data.head())

st.header('Normalizacion de datos')
features = data.iloc[:, :-1].values
targets = data.iloc[:, -1].values
features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
targets = (targets - np.mean(targets)) / np.std(targets)
st.write('datos normalizados')

class redNeuronal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(redNeuronal, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

st.header("creando la Red Neuronal")
input_size = features.shape[1]
hidden_size = 64
output_size = 1
model = redNeuronal(input_size, hidden_size, output_size)
st.write("red neuronal creada.")

st.header("entrenamiento de la ia")
features_tensor = torch.tensor(features, dtype=torch.float32)
targets_tensor = torch.tensor(targets, dtype=torch.float32).view(-1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 500
losses = []
for epoch in range(epochs):
    predictions = model(features_tensor)
    loss = criterion(predictions, targets_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        st.write(f"ciclo [{epoch + 1}/{epochs}], perdida: {loss.item():.4f}")

torch.save(model.state_dict(), "modelo_ventas.pth")
st.write("modelo guardado como 'modelo_ventas.pth'.")

st.header("grafica de las prediccion ")
predictions = model(features_tensor).detach().numpy()
plt.figure(figsize=(10, 5))
plt.plot(targets, label="real")
plt.plot(predictions, label="prediccion")
plt.legend()
plt.title("comparacion entre ventas reales y predicciones")
st.pyplot(plt)