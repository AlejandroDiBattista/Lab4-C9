import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59059.streamlit.app'


st.set_page_config(layout="wide", page_title="Análisis de Ventas", page_icon=":bar_chart:")

st.markdown(
    """
    <style>
    .stApp, .stAppHeader{
        background-color: white;
    }
    .stApp, .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: black !important;
    }
    div[data-testid="stMetricValue"] > div {
        color: black !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricLabel"] > label {
        color: black !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetricDelta"] > div {
        font-weight: bold !important;
    }
    div[data-testid="stMetricDelta"] svg {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def mostrar_informacion_alumno():
    st.markdown('**Legajo:** 59059')
    st.markdown('**Nombre:** Soraire Elias Nicolas')
    st.markdown('**Comisión:** C9')

def cargar_datos(archivo):
    df = pd.read_csv(archivo)
    df['Fecha'] = df['Año'].astype(str) + '-' + df['Mes'].astype(str).str.zfill(2)
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y-%m')
    return df

def calcular_metricas(df, producto):
    df_producto = df[df['Producto'] == producto]
    precio_promedio = df_producto['Ingreso_total'].sum() / df_producto['Unidades_vendidas'].sum()
    margen_promedio = ((df_producto['Ingreso_total'].sum() - df_producto['Costo_total'].sum()) / df_producto['Ingreso_total'].sum()) * 100
    unidades_vendidas = df_producto['Unidades_vendidas'].sum()
    
    df_producto = df_producto.sort_values('Fecha')
    precio_cambio = (df_producto['Ingreso_total'].iloc[-1] / df_producto['Unidades_vendidas'].iloc[-1] - df_producto['Ingreso_total'].iloc[0] / df_producto['Unidades_vendidas'].iloc[0]) / (df_producto['Ingreso_total'].iloc[0] / df_producto['Unidades_vendidas'].iloc[0]) * 100
    margen_cambio = ((df_producto['Ingreso_total'].iloc[-1] - df_producto['Costo_total'].iloc[-1]) / df_producto['Ingreso_total'].iloc[-1] - (df_producto['Ingreso_total'].iloc[0] - df_producto['Costo_total'].iloc[0]) / df_producto['Ingreso_total'].iloc[0]) / ((df_producto['Ingreso_total'].iloc[0] - df_producto['Costo_total'].iloc[0]) / df_producto['Ingreso_total'].iloc[0]) * 100
    unidades_cambio = (df_producto['Unidades_vendidas'].iloc[-1] - df_producto['Unidades_vendidas'].iloc[0]) / df_producto['Unidades_vendidas'].iloc[0] * 100
    
    return precio_promedio, margen_promedio, unidades_vendidas, precio_cambio, margen_cambio, unidades_cambio

def mostrar_grafico(df, producto):
    df_producto = df[df['Producto'] == producto]
    df_producto = df_producto.sort_values('Fecha')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_producto['Fecha'], df_producto['Unidades_vendidas'], label=producto)
    
    z = np.polyfit(range(len(df_producto)), df_producto['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(df_producto['Fecha'], p(range(len(df_producto))), "r--", label='Tendencia')
    
    ax.set_xlabel('Año-Mes')
    ax.set_ylabel('Unidades Vendidas')
    ax.set_title(f'Evolución de Ventas Mensual')
    ax.legend()
    ax.grid(True)
    
    plt.xticks(rotation=45)
    
    return fig

def mostrar_metrica_personalizada(label, value, delta):
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 14px; color: black; font-weight: bold;">{label}</div>
        <div style="font-size: 24px; color: black; font-weight: bold;">{value}</div>
        <div style="font-size: 14px; color: {'green' if float(delta.rstrip('%')) > 0 else 'red'}; font-weight: bold;">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.title('Cargar archivo de datos')
    archivo = st.file_uploader('Subir archivo CSV', type='csv')
    
    if archivo is not None:
        df = cargar_datos(archivo)
        sucursal = st.selectbox('Seleccionar Sucursal', ['Todas'] + sorted(df['Sucursal'].unique().tolist()))

with col2:
    st.title("Por favor, sube un archivo CSV desde la barra lateral.")
    with st.container(border=True):
        mostrar_informacion_alumno()

    if archivo is not None:
        if sucursal != 'Todas':
            df = df[df['Sucursal'] == sucursal]
        
        st.header(f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}")
        
        for producto in sorted(df['Producto'].unique()):
            with st.container(border=True):
                col1, col2 = st.columns([1, 2])
                
                precio_promedio, margen_promedio, unidades_vendidas, precio_cambio, margen_cambio, unidades_cambio = calcular_metricas(df, producto)
                
                with col1:
                    st.subheader(producto)
                    mostrar_metrica_personalizada("Precio Promedio", f"${precio_promedio:.2f}", f"{precio_cambio:.2f}%")
                    mostrar_metrica_personalizada("Margen Promedio", f"{margen_promedio:.2f}%", f"{margen_cambio:.2f}%")
                    mostrar_metrica_personalizada("Unidades Vendidas", f"{unidades_vendidas:,}", f"{unidades_cambio:.2f}%")
                
                with col2:
                    fig = mostrar_grafico(df, producto)
                    st.pyplot(fig)