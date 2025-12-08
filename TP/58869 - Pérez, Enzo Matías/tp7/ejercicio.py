import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List


st.set_page_config(
    page_title="Estimación de Ventas Diarias",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
        .main > div {
            padding: 2rem 1rem;
        }
        .stButton>button {
            width: 100%;
        }
        div.block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Parámetros de Entrenamiento")
    
    learning_rate = st.number_input(
        "Aprendizaje",
        min_value=0.0,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.4f"
    )
    
    st.markdown("##")
    
    epochs = st.number_input(
        "Repeticiones",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10
    )
    
    st.markdown("##")
    
    st.subheader("Neuronas Capa Oculta")
    hidden_size = st.number_input(
        "",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        label_visibility="collapsed"
    )
    
    st.markdown("##")
    
    train_button = st.button("Entrenar")
    
  
    progress_container = st.empty()
    
   
    success_container = st.empty()
    
 
    loss_plot_container = st.empty()


st.title("Estimación de Ventas Diarias")


class VentasNet(nn.Module):
    def __init__(self, hidden_size: int):
        super(VentasNet, self).__init__()
        self.hidden = nn.Linear(1, hidden_size)
        self.act = nn.ReLU()
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.act(x)
        x = self.output(x)
        return x


class VentasDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    progress_container
) -> List[float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    progress_bar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Epoch {epoch + 1}/{epochs} - Error: {epoch_loss:.6f}')
        losses.append(epoch_loss)
    
    return losses


@st.cache_data
def load_data():
    df = pd.read_csv('ventas.csv')
    return df


df = load_data()
X = df['dia'].values.reshape(-1, 1)
y = df['ventas'].values.reshape(-1, 1)


from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)


dataset = VentasDataset(X_scaled, y_scaled)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


plot_container = st.container()

if train_button:

    model = VentasNet(hidden_size)
    losses = train_model(model, train_loader, learning_rate, epochs, progress_container)
    

    success_container.success("Entrenamiento exitoso")

    with loss_plot_container:
        fig_loss, ax_loss = plt.subplots(figsize=(6, 4))
        ax_loss.plot(losses, color='green', label='Pérdidas')
        ax_loss.set_xlabel('Época')
        ax_loss.set_ylabel('Pérdida')
        ax_loss.grid(True)
        st.pyplot(fig_loss)
    
    with plot_container:
       
        model.eval()
        with torch.no_grad():
            X_test = torch.FloatTensor(X_scaled)
            y_pred_scaled = model(X_test)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
        
      
        fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
        ax_pred.scatter(X, y, color='blue', label='Datos Reales')
        ax_pred.plot(X, y_pred, color='red', label='Curva de Ajuste')
        ax_pred.set_xlabel('Día del Mes')
        ax_pred.set_ylabel('Ventas')
        ax_pred.legend()
        ax_pred.grid(True)
        st.pyplot(fig_pred)
else:
   
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Datos Reales')
    plt.xlabel('Día del Mes')
    plt.ylabel('Ventas')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)