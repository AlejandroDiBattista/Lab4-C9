import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


st.set_page_config(page_title="Predicción de Ventas - Red Neuronal", layout="wide")


class VentasDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RedNeuronal(nn.Module):
    def __init__(self, neuronas_ocultas):
        super(RedNeuronal, self).__init__()
        self.hidden = nn.Linear(1, neuronas_ocultas)
        self.act = nn.ReLU()
        self.output = nn.Linear(neuronas_ocultas, 1)
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.act(x)
        x = self.output(x)
        return x


def entrenar_red(modelo, criterion, optimizer, train_loader, epochs, progress_bar):
    losses = []
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            y_pred = modelo(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        progress_bar.progress((epoch + 1) / epochs)
    return losses


@st.cache_data
def cargar_datos():
    data = {
        'dia': list(range(1, 31)),
        'ventas': [195,169,172,178,132,123,151,127,96,110,86,82,94,60,63,76,69,98,77,71,134,107,120,99,126,150,136,179,173,194]
    }
    df = pd.DataFrame(data)
    X = df['dia'].values.reshape(-1, 1)
    y = df['ventas'].values.reshape(-1, 1)
    return X, y, df


with st.sidebar:
    st.header('Parámetros de Entrenamiento')
    learning_rate = st.number_input('Aprendizaje', 0.0, 0.1, 0.01, step=0.0001, format="%.4f")
    epochs = st.number_input('Repeticiones', 10, 10000, 1000, step=10)
    hidden_neurons = st.number_input('Neuronas Capa Oculta', 1, 100, 10, step=1)
    entrenar = st.button('Entrenar')


X, y, df = cargar_datos()
X_norm = (X - X.mean()) / X.std()
y_norm = (y - y.mean()) / y.std()
dataset = VentasDataset(X_norm, y_norm)
train_loader = DataLoader(dataset, batch_size=len(X), shuffle=False)

if entrenar:
    modelo = RedNeuronal(hidden_neurons)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    
    progress_bar = st.progress(0)
    losses = entrenar_red(modelo, criterion, optimizer, train_loader, epochs, progress_bar)
    
    st.success('Entrenamiento exitoso')
    
    
    with st.sidebar:
        fig_loss, ax_loss = plt.subplots(figsize=(3, 2))
        ax_loss.plot(losses, color='green')
        ax_loss.set_xlabel('Época')
        ax_loss.set_ylabel('Pérdida')
        ax_loss.grid(True)
        plt.tight_layout()
        st.pyplot(fig_loss)
        st.text(f'Epoch {epochs}/{epochs} - Error: {losses[-1]:.5f}')

    
    modelo.eval()
    with torch.no_grad():
        X_pred = np.linspace(0, 30, 100).reshape(-1, 1)
        X_pred_norm = (X_pred - X.mean()) / X.std()
        X_pred_tensor = torch.FloatTensor(X_pred_norm)
        y_pred_norm = modelo(X_pred_tensor)
        y_pred = y_pred_norm.numpy() * y.std() + y.mean()


fig, ax = plt.subplots()  
ax.scatter(X, y, color='blue', label='Datos Reales', s=30)
ax.set_xlabel('Día del Mes')
ax.set_ylabel('Ventas')
ax.set_title('Estimación de Ventas Diarias')
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xlim(0, 30)
ax.set_ylim(50, 210)

if entrenar:
    ax.plot(X_pred, y_pred, 'r-', label='Curva de Ajuste', linewidth=2)
    ax.legend()

plt.tight_layout()
st.pyplot(fig)