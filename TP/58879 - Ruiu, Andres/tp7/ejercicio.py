import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

st.set_page_config(page_title="Estimación de Ventas", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('ventas.csv')

data = load_data()

class SalesEstimator(nn.Module):
    def __init__(self, input_size, num_hidden_neurons, output_size):
        super(SalesEstimator, self).__init__()
        self.fc1 = nn.Linear(input_size, num_hidden_neurons)
        self.fc2 = nn.Linear(num_hidden_neurons, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data - mean) / std
    return normalized, mean, std

def denormalize_data(normalized_data, mean, std):
    return normalized_data * std + mean

def prepare_training_data(data):
    X = data['dia'].values.reshape(-1, 1).astype(np.float32)
    y = data['ventas'].values.reshape(-1, 1).astype(np.float32)
    
    X_normalized, X_mean, X_std = normalize_data(X)
    y_normalized, y_mean, y_std = normalize_data(y)
    
    X_tensor = torch.from_numpy(X_normalized)
    y_tensor = torch.from_numpy(y_normalized)
    
    normalization_params = {
        'X_mean': X_mean,
        'X_std': X_std,
        'y_mean': y_mean,
        'y_std': y_std
    }
    
    return X_tensor, y_tensor, normalization_params

def train_model(learning_rate, num_epochs, num_hidden_neurons, early_stopping_patience=10):
    model = SalesEstimator(input_size=1, num_hidden_neurons=num_hidden_neurons, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    X_tensor, y_tensor, norm_params = prepare_training_data(data)
    
    train_losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    with st.container():
        progress_bar = st.progress(0, text='Entrenando modelo...')
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            train_losses.append(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                st.info(f"Entrenamiento detenido tempranamente en epoch {epoch + 1} debido a no mejora")
                break
                
            progress_bar.progress((epoch + 1) / num_epochs, 
                                text=f'Epoch {epoch + 1}/{num_epochs} - Error: {current_loss:.6f}')

    for key, value in norm_params.items():
        setattr(model, key, value)

    return model, train_losses

def predict_sales(model, X):
    X_normalized = (X - model.X_mean) / model.X_std
    X_tensor = torch.from_numpy(X_normalized.astype(np.float32))
    
    with torch.no_grad():
        predictions = model(X_tensor)
    
    predictions_np = predictions.numpy()
    return denormalize_data(predictions_np, model.y_mean, model.y_std)

def plot_sales_data(model=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    days = data['dia'].values
    sales = data['ventas'].values
    
    ax.scatter(days, sales, color='blue', label='Datos Reales', alpha=0.6)
    
    if model is not None:
        X = days.reshape(-1, 1)
        predictions = predict_sales(model, X)
        ax.plot(days, predictions, color='red', label='Curva de Ajuste', linewidth=2)
    
    ax.set_xlabel('Día del Mes')
    ax.set_ylabel('Ventas')
    ax.set_title('Estimación de Ventas Diarias')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    st.pyplot(fig)

def plot_loss_curve(train_losses):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    losses = np.array(train_losses)
    window_size = max(len(losses) // 20, 1)
    smoothed_losses = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
    
    ax.plot(losses, color='lightgreen', alpha=0.3, label='Pérdida Original')
    ax.plot(np.arange(window_size-1, len(losses)), smoothed_losses, 
            color='green', linewidth=2, label='Pérdida Suavizada')
    
    ax.set_xlabel('Época')
    ax.set_ylabel('Pérdida')
    ax.set_title('Evolución de la Función de Costo')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    st.sidebar.pyplot(fig)

st.title('Estimación de Ventas Diarias')

initial_plot_container = st.container()

st.sidebar.header('Parámetros de Entrenamiento')

col1, col2 = st.sidebar.columns(2)
with col1:
    learning_rate = st.number_input('Tasa de Aprendizaje', min_value=0.0001, max_value=1.0, 
                                  value=0.1, step=0.0001, format="%.4f")
with col2:
    num_epochs = st.number_input('Épocas', min_value=10, max_value=10000, value=100, step=10)

num_hidden_neurons = st.sidebar.number_input('Neuronas Capa Oculta', min_value=1, max_value=100, value=5, step=1)

trained_plot_container = st.container()

if st.sidebar.button('Entrenar Modelo'):
    initial_plot_container.empty()
    
    with st.sidebar.container():
        model, train_losses = train_model(learning_rate, num_epochs, num_hidden_neurons)
        st.success('¡Entrenamiento exitoso!')
    
    with trained_plot_container:
        plot_sales_data(model)
        plot_loss_curve(train_losses)
else:
    with initial_plot_container:
        plot_sales_data()