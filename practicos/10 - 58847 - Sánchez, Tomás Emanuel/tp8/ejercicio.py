import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import requests

def configurar_pagina():
    st.set_page_config(
        page_title="Análisis de Ventas",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stMetric {
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        .stMetric label {
            color: #555;
        }
        .css-1l02zno {
            background-color: #f5f5f5;
            border-right: 1px solid #ddd;
        }
        </style>
    """, unsafe_allow_html=True)

def mostrar_informacion_alumno():
    st.markdown("""
        <div style='background-color: white; padding: 1.5rem; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.12);'>
            <p><strong>Legajo:</strong> 58847</p>
            <p><strong>Nombre:</strong> Sánchez, Tomás Emanuel</p>
            <p><strong>Comisión:</strong> C9</p>
        </div>
    """, unsafe_allow_html=True)

def cargar_datos(url=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif url:
        response = requests.get(url)
        df = pd.read_csv(StringIO(response.text))
    else:
        return None
    
    if 'AÃ±o' in df.columns:
        df = df.rename(columns={'AÃ±o': 'Año'})
    return df

def calcular_metricas_con_cambios(df, periodo_anterior=12):
    metricas = {}
    for producto in df['Producto'].unique():
        df_producto = df[df['Producto'] == producto]
        
        precio_promedio = df_producto['Ingreso_total'].sum() / df_producto['Unidades_vendidas'].sum()
        margen_promedio = (df_producto['Ingreso_total'].sum() - df_producto['Costo_total'].sum()) / df_producto['Ingreso_total'].sum()
        unidades_vendidas = df_producto['Unidades_vendidas'].sum()
        
        df_anterior = df_producto.iloc[:-periodo_anterior] if len(df_producto) > periodo_anterior else pd.DataFrame()
        if not df_anterior.empty:
            precio_anterior = df_anterior['Ingreso_total'].sum() / df_anterior['Unidades_vendidas'].sum()
            margen_anterior = (df_anterior['Ingreso_total'].sum() - df_anterior['Costo_total'].sum()) / df_anterior['Ingreso_total'].sum()
            unidades_anterior = df_anterior['Unidades_vendidas'].sum()
            
            cambio_precio = ((precio_promedio - precio_anterior) / precio_anterior) * 100
            cambio_margen = ((margen_promedio - margen_anterior) / margen_anterior) * 100
            cambio_unidades = ((unidades_vendidas - unidades_anterior) / unidades_anterior) * 100
        else:
            cambio_precio = cambio_margen = cambio_unidades = 0
        
        metricas[producto] = {
            'precio_promedio': precio_promedio,
            'margen_promedio': margen_promedio,
            'unidades_vendidas': unidades_vendidas,
            'cambio_precio': cambio_precio,
            'cambio_margen': cambio_margen,
            'cambio_unidades': cambio_unidades
        }
    
    return metricas

def graficar_evolucion_ventas(df, producto):
    df_producto = df[df['Producto'] == producto].copy()
    df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + 
                                          df_producto['Mes'].astype(str).str.zfill(2) + '-01')
    df_producto = df_producto.sort_values('Fecha')
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    ax.plot(df_producto['Fecha'], df_producto['Unidades_vendidas'], 
            label=producto, color='blue', linewidth=1.5)
    
    z = np.polyfit(range(len(df_producto)), df_producto['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(df_producto['Fecha'], p(range(len(df_producto))), 
            '--', color='red', label='Tendencia', linewidth=1.5)
    
    ax.set_title('Evolución de Ventas Mensual', pad=20)
    ax.set_xlabel('Año-Mes')
    ax.set_ylabel('Unidades Vendidas')
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def mostrar_metricas_producto(df, producto, metricas):
    st.subheader(producto)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Precio Promedio",
            value=f"${metricas[producto]['precio_promedio']:,.2f}",
            delta=f"{metricas[producto]['cambio_precio']:+.2f}%"
        )
    
    with col2:
        st.metric(
            label="Margen Promedio",
            value=f"{metricas[producto]['margen_promedio']:.1%}",
            delta=f"{metricas[producto]['cambio_margen']:+.2f}%"
        )
    
    with col3:
        st.metric(
            label="Unidades Vendidas",
            value=f"{int(metricas[producto]['unidades_vendidas']):,}",
            delta=f"{metricas[producto]['cambio_unidades']:+.2f}%"
        )
    
    fig = graficar_evolucion_ventas(df, producto)
    st.pyplot(fig)

def main():
    configurar_pagina()
    
    with st.sidebar:
        st.header("Cargar archivo de datos")
        st.subheader("Subir archivo CSV")
        uploaded_file = st.file_uploader(
            "Drag and drop file here",
            type="csv",
            help="Limit 200MB per file • CSV"
        )
        
        if not uploaded_file:
            st.info("Por favor, sube un archivo CSV desde la barra lateral.")
            mostrar_informacion_alumno()
            return
        
        df = cargar_datos(uploaded_file=uploaded_file)
        if df is None:
            st.error("Error al cargar el archivo")
            return
        
        
            
        st.subheader("Seleccionar Sucursal")
        sucursales = ['Todas'] + list(df['Sucursal'].unique())
        sucursal_seleccionada = st.selectbox('Sucursal', sucursales)
    
    if sucursal_seleccionada != 'Todas':
        st.title(f"Datos de {sucursal_seleccionada}")
        df_filtrado = df[df['Sucursal'] == sucursal_seleccionada]
    else:
        st.title("Datos de Todas las Sucursales")
        df_filtrado = df
    
    metricas = calcular_metricas_con_cambios(df_filtrado)
    
    for producto in df_filtrado['Producto'].unique():
        with st.container():
            mostrar_metricas_producto(df_filtrado, producto, metricas)
            st.markdown("---")

if __name__ == '__main__':
    main()