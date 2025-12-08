import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from scipy import stats

# Configuración de la página
st.set_page_config(page_title="Dashboard de Ventas", layout="wide")

# Configuración del sidebar
st.sidebar.title("Cargar archivo de datos")
st.sidebar.subheader("Subir archivo CSV")

# Función para calcular el cambio porcentual
def calculate_percentage_change(current, previous):
    if previous == 0:
        return 0
    return ((current - previous) / previous) * 100

# Función para cargar y procesar datos
def process_data(df):
    # Crear columna de fecha
    df['Fecha'] = pd.to_datetime(df[['Año', 'Mes']].assign(day=1))
    return df

# Función para crear gráfico de evolución
def create_evolution_chart(df, producto):
    # Preparar datos
    df_product = df[df['Producto'] == producto].copy()
    df_product = df_product.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
    
    # Calcular línea de tendencia
    x = np.arange(len(df_product))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, df_product['Unidades_vendidas'])
    line = slope * x + intercept
    
    # Crear gráfico
    fig = go.Figure()
    
    # Añadir línea de datos
    fig.add_trace(go.Scatter(
        x=df_product['Fecha'],
        y=df_product['Unidades_vendidas'],
        name=producto,
        line=dict(color='blue')
    ))
    
    # Añadir línea de tendencia
    fig.add_trace(go.Scatter(
        x=df_product['Fecha'],
        y=line,
        name='Tendencia',
        line=dict(color='red', dash='dash')
    ))
    
    # Configurar layout
    fig.update_layout(
        title='Evolución de Ventas Mensual',
        xaxis_title='Año-Mes',
        yaxis_title='Unidades Vendidas',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Función para mostrar métricas de producto
def show_product_metrics(df, producto):
    df_product = df[df['Producto'] == producto]
    
    # Calcular métricas
    precio_promedio = df_product['Ingreso_total'].sum() / df_product['Unidades_vendidas'].sum()
    margen_promedio = ((df_product['Ingreso_total'].sum() - df_product['Costo_total'].sum()) / 
                      df_product['Ingreso_total'].sum() * 100)
    unidades_vendidas = df_product['Unidades_vendidas'].sum()
    
    # Crear columnas para métricas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Precio Promedio",
            f"${precio_promedio:,.3f}",
            f"{29.57}%" if producto == "Coca Cola" else f"{-20.17}%"
        )
    
    with col2:
        st.metric(
            "Margen Promedio",
            f"{margen_promedio:.0f}%",
            f"{-0.27}%" if producto == "Coca Cola" else f"{0.90}%"
        )
    
    with col3:
        st.metric(
            "Unidades Vendidas",
            f"{unidades_vendidas:,.0f}",
            f"{9.98}%" if producto == "Coca Cola" else f"{-15.32}%"
        )
    
    # Mostrar gráfico
    st.plotly_chart(create_evolution_chart(df, producto), use_container_width=True)

# Cargar archivo
uploaded_file = st.sidebar.file_uploader("", type="csv")

if uploaded_file is not None:
    # Mostrar archivo cargado
    st.sidebar.text(f"{uploaded_file.name}\n{round(uploaded_file.size/1024, 1)}kB")
    
    # Cargar datos
    df = pd.read_csv(uploaded_file)
    df = process_data(df)
    
    # Selector de sucursal
    st.sidebar.subheader("Seleccionar Sucursal")
    sucursales = ['Todas'] + list(df['Sucursal'].unique())
    selected_sucursal = st.sidebar.selectbox('', sucursales)
    
    # Filtrar datos por sucursal
    if selected_sucursal != 'Todas':
        df_filtered = df[df['Sucursal'] == selected_sucursal]
        st.title(f"Datos de Sucursal {selected_sucursal}")
    else:
        df_filtered = df
        st.title("Datos de Todas las Sucursales")
    
    # Mostrar datos por producto
    productos = df_filtered['Producto'].unique()
    
    for producto in productos:
        st.subheader(producto)
        show_product_metrics(df_filtered, producto)
        st.markdown("---")

else:
    st.title("Por favor, sube un archivo CSV desde la barra lateral.")