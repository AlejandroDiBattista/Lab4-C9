import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Análisis de Ventas", page_icon=":bar_chart:")

# Agregar CSS mínimo para el contenedor y colores
st.markdown("""
    <style>
    .product-container {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

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
    df_producto['Precio_promedio'] = df_producto['Ingreso_total'] / df_producto['Unidades_vendidas']
    df_producto['Margen_promedio'] = (df_producto['Ingreso_total'] - df_producto['Costo_total']) / df_producto['Ingreso_total']

    precio_promedio = df_producto['Precio_promedio'].mean()
    margen_promedio = df_producto['Margen_promedio'].mean() * 100
    unidades_vendidas = df_producto['Unidades_vendidas'].sum()

    precio_cambio = (df_producto['Precio_promedio'].pct_change().mean()) * 100
    margen_cambio = (df_producto['Margen_promedio'].pct_change().mean())
    unidades_cambio = (df_producto['Unidades_vendidas'].pct_change().mean()) * 100

    return precio_promedio, margen_promedio, unidades_vendidas, precio_cambio, margen_cambio, unidades_cambio

def mostrar_grafico(df, producto):
    df_producto = df[df['Producto'] == producto]
    df_producto = df_producto.sort_values('Fecha')
    
    # Configurar el estilo del gráfico con fondo blanco
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Graficar con colores más visibles
    ax.plot(df_producto['Fecha'], df_producto['Unidades_vendidas'], color='#0066cc', label=producto)
    
    z = np.polyfit(range(len(df_producto)), df_producto['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(df_producto['Fecha'], p(range(len(df_producto))), color='#cc0000', linestyle='--', label='Tendencia')
    
    ax.set_xlabel('Año-Mes')
    ax.set_ylabel('Unidades Vendidas')
    ax.set_title(f'Evolución de Ventas Mensual')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def mostrar_metrica_personalizada(label, value, delta):
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <div style="font-size: 14px; color: white; font-weight: bold;">{label}</div>
        <div style="font-size: 24px; color: white; font-weight: bold;">{value}</div>
        <div style="font-size: 14px; color: {'#00ff00' if float(delta.rstrip('%')) > 0 else '#ff0000'}; font-weight: bold;">{delta}</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.title('Cargar archivo de datos')
archivo = st.sidebar.file_uploader('Subir archivo CSV', type='csv')

if archivo is None:
    st.title("Por favor, sube un archivo CSV desde la barra lateral.")
    with st.container():
        mostrar_informacion_alumno()
else:
    df = cargar_datos(archivo)
    sucursal = st.sidebar.selectbox('Seleccionar Sucursal', ['Todas'] + sorted(df['Sucursal'].unique().tolist()))

    if sucursal != 'Todas':
        df = df[df['Sucursal'] == sucursal]
    
    st.title(f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}")
    
    for producto in sorted(df['Producto'].unique()):
        st.markdown('<div class="product-container">', unsafe_allow_html=True)
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
        
        st.markdown('</div>', unsafe_allow_html=True)