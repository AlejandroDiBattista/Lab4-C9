import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import requests
utl="https://t8-lourdesrocha.streamlit.app/"

def setup_page():
    st.set_page_config(
        page_title="Análisis de Ventas Avanzado",
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
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .stMetric label {
            color: #2c3e50;
            font-weight: bold;
        }
        .css-1l02zno {
            border-right: 1px solid #e9ecef;
        }
        </style>
    """, unsafe_allow_html=True)

def show_student_info():
    st.sidebar.markdown("""
        <div style='background-color: #34495e; color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);'>
            <h3 style='margin-bottom: 10px;'>Información del Estudiante</h3>
            <p><strong>Legajo:</strong> 59057</p>
            <p><strong>Nombre:</strong> Rocha Lourdes Gabriela</p>
            <p><strong>Comisión:</strong> 9</p>
        </div>
    """, unsafe_allow_html=True)

def load_data(uploaded_file=None, url=None):
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

def compute_metrics(df, lookback_period=12):
    metrics = {}
    for product in df['Producto'].unique():
        df_product = df[df['Producto'] == product]
        
        avg_price = df_product['Ingreso_total'].sum() / df_product['Unidades_vendidas'].sum()
        avg_margin = (df_product['Ingreso_total'].sum() - df_product['Costo_total'].sum()) / df_product['Ingreso_total'].sum()
        total_units = df_product['Unidades_vendidas'].sum()
        
        df_previous = df_product.iloc[:-lookback_period] if len(df_product) > lookback_period else pd.DataFrame()
        if not df_previous.empty:
            prev_price = df_previous['Ingreso_total'].sum() / df_previous['Unidades_vendidas'].sum()
            prev_margin = (df_previous['Ingreso_total'].sum() - df_previous['Costo_total'].sum()) / df_previous['Ingreso_total'].sum()
            prev_units = df_previous['Unidades_vendidas'].sum()
            
            price_delta = ((avg_price - prev_price) / prev_price) * 100
            margin_delta = ((avg_margin - prev_margin) / prev_margin) * 100
            units_delta = ((total_units - prev_units) / prev_units) * 100
        else:
            price_delta = margin_delta = units_delta = 0
        
        metrics[product] = {
            'avg_price': avg_price,
            'avg_margin': avg_margin,
            'total_units': total_units,
            'price_delta': price_delta,
            'margin_delta': margin_delta,
            'units_delta': units_delta
        }
    
    return metrics

def plot_sales_trend(df, product):
    df_product = df[df['Producto'] == product].copy()
    df_product['Fecha'] = pd.to_datetime(df_product['Año'].astype(str) + '-' + 
                                         df_product['Mes'].astype(str).str.zfill(2) + '-01')
    df_product = df_product.sort_values('Fecha')
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    
    ax.plot(df_product['Fecha'], df_product['Unidades_vendidas'], 
            label=product, color='#3498db', linewidth=2)
    
    z = np.polyfit(range(len(df_product)), df_product['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(df_product['Fecha'], p(range(len(df_product))), 
            '--', color='#e74c3c', label='Tendencia', linewidth=2)
    
    ax.set_title(f'Tendencia de Ventas - {product}', fontsize=16, pad=20)
    ax.set_xlabel('Fecha', fontsize=12)
    ax.set_ylabel('Unidades Vendidas', fontsize=12)
    ax.legend(fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig

def display_product_analysis(df, product, metrics):
    st.header(f"Análisis de {product}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Precio Promedio",
            value=f"${metrics[product]['avg_price']:,.2f}",
            delta=f"{metrics[product]['price_delta']:+.2f}%"
        )
    
    with col2:
        st.metric(
            label="Margen Promedio",
            value=f"{metrics[product]['avg_margin']:.1%}",
            delta=f"{metrics[product]['margin_delta']:+.2f}%"
        )
    
    with col3:
        st.metric(
            label="Total Unidades Vendidas",
            value=f"{int(metrics[product]['total_units']):,}",
            delta=f"{metrics[product]['units_delta']:+.2f}%"
        )
    
    fig = plot_sales_trend(df, product)
    st.pyplot(fig)

def main():
    setup_page()
    
    st.sidebar.title("Carga de Datos")
    uploaded_file = st.sidebar.file_uploader(
        "Subir archivo CSV",
        type="csv",
        help="Tamaño máximo: 200MB por archivo"
    )
    
    if not uploaded_file:
        st.warning("Por favor, sube un archivo CSV desde la barra lateral.")
        show_student_info()
        return
    
    df = load_data(uploaded_file=uploaded_file)
    if df is None:
        st.error("Error al cargar el archivo")
        return
    
    st.sidebar.title("Opciones de Filtrado")
    branches = ['Todas'] + list(df['Sucursal'].unique())
    selected_branch = st.sidebar.selectbox('Seleccionar Sucursal', branches)
    
    if selected_branch != 'Todas':
        st.title(f"Análisis de Ventas - {selected_branch}")
        df_filtered = df[df['Sucursal'] == selected_branch]
    else:
        st.title("Análisis de Ventas - Todas las Sucursales")
        df_filtered = df
    
    metrics = compute_metrics(df_filtered)
    
    for product in df_filtered['Producto'].unique():
        with st.expander(f"Detalles de {product}", expanded=True):
            display_product_analysis(df_filtered, product, metrics)
            st.markdown("---")

if __name__ == '__main__':
    main()