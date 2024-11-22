import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


st.set_page_config(page_title="Predicción de Ventas", layout="wide")

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
        self.capa1 = nn.Linear(1, neuronas_ocultas)
        self.activacion = nn.ReLU()
        self.capa2 = nn.Linear(neuronas_ocultas, 1)
    
    def forward(self, x):
        x = self.capa1(x)
        x = self.activacion(x)
        x = self.capa2(x)
        return x


@st.cache_data
def cargar_datos():
    df = pd.read_csv('ventas.csv')
    X = df['dia'].values.reshape(-1, 1)
    y = df['ventas'].values.reshape(-1, 1)
    
    
    X = (X - X.mean()) / X.std()
    y = (y - y.mean()) / y.std()
    
    return df, X, y, y.mean(), y.std()


def entrenar_red(X, y, neuronas_ocultas, learning_rate, epochs):
    
    dataset = VentasDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    
    modelo = RedNeuronal(neuronas_ocultas)
    criterio = nn.MSELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    
    
    perdidas = []
    
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            
            predicciones = modelo(batch_X)
            loss = criterio(predicciones, batch_y)
            
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        perdidas.append(loss.item())
        
        
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f'Época {epoch + 1}/{epochs} - Error: {loss.item():.6f}')
    
    return modelo, perdidas

def main():
    # Título
    st.title("Estimación de Ventas Diarias")
    

    col1, col2 = st.columns([1, 3])
    
    
    with col1:
        st.header("Parámetros de Entrenamiento")
        learning_rate = st.number_input("Aprendizaje", 0.0, 1.0, 0.01, 0.001)
        epochs = st.number_input("Repeticiones", 10, 10000, 1000, 10)
        neuronas = st.number_input("Neuronas Capa Oculta", 1, 100, 10, 1)
        
        entrenar = st.button("Entrenar")
    
  
    df, X, y, y_mean, y_std = cargar_datos()
    
    
    with col2:
        if entrenar:
            
            modelo, perdidas = entrenar_red(X, y, neuronas, learning_rate, epochs)
            
            
            st.success("Entrenamiento exitoso")
            
            
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(perdidas, 'g-', label='Pérdidas')
            ax_loss.set_xlabel('Época')
            ax_loss.set_ylabel('Pérdida')
            ax_loss.legend()
            st.pyplot(fig_loss)
            
           
            modelo.eval()
            with torch.no_grad():
                X_test = torch.FloatTensor(X)
                predicciones = modelo(X_test).numpy()
            
            
            predicciones = predicciones * y_std + y_mean
            
            
            fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
            ax_pred.scatter(df['dia'], df['ventas'], color='blue', label='Datos Reales')
            ax_pred.plot(df['dia'], predicciones, 'r-', label='Curva de Ajuste')
            ax_pred.set_xlabel('Día del Mes')
            ax_pred.set_ylabel('Ventas')
            ax_pred.legend()
            st.pyplot(fig_pred)
        else:
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df['dia'], df['ventas'], color='blue', label='Datos Reales')
            ax.set_xlabel('Día del Mes')
            ax.set_ylabel('Ventas')
            ax.legend()
            st.pyplot(fig)

if __name__ == '__main__':
    main()