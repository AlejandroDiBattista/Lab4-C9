import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, YearLocator


def cargar_datos(archivo_csv):
    if archivo_csv is not None:
        return pd.read_csv(archivo_csv)
    return None


def calc_metricas(df, prod=None):
    df = df.copy()
    df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str), format='%Y-%m')
    
    df['Precio'] = df['Ingreso_total'].div(df['Unidades_vendidas'])
    df['Margen'] = ((df['Ingreso_total'] - df['Costo_total'])
                    .div(df['Ingreso_total'])
                    .fillna(0) * 100)
    
    df_prod = df[df['Producto'] == prod] if prod else df
    
    def calc_periodo(data):
        return pd.DataFrame({
            'Precio': [data['Precio'].mean()],
            'Margen': [data['Margen'].mean()],
            'Unidades': [data['Unidades_vendidas'].sum()]
        })
    
    metrics = calc_periodo(df_prod)
    
    años = sorted(df_prod['Año'].unique())
    
    if len(años) > 1:
        año_act = años[-1]
        año_ant = años[-2]
        
        metrics_act = calc_periodo(df_prod[df_prod['Año'] == año_act])
        metrics_ant = calc_periodo(df_prod[df_prod['Año'] == año_ant])
        
        variaciones = pd.DataFrame({
            'var_precio': [(metrics_act['Precio'].iloc[0] / 
                           metrics_ant['Precio'].iloc[0] - 1) * 100 
                          if metrics_ant['Precio'].iloc[0] != 0 else 0],
            
            'var_margen': [metrics_act['Margen'].iloc[0] - 
                          metrics_ant['Margen'].iloc[0]],
            
            'var_unidades': [(metrics_act['Unidades'].iloc[0] / 
                             metrics_ant['Unidades'].iloc[0] - 1) * 100
                            if metrics_ant['Unidades'].iloc[0] != 0 else 0]
        })
    else:
        variaciones = pd.DataFrame({
            'var_precio': [0],
            'var_margen': [0],
            'var_unidades': [0]
        })
    
    return (
        metrics['Precio'].iloc[0],
        variaciones['var_precio'].iloc[0],
        metrics['Margen'].iloc[0],
        variaciones['var_margen'].iloc[0],
        metrics['Unidades'].iloc[0],
        variaciones['var_unidades'].iloc[0]
    )


def calc_tendencia(x, y):
    return np.polyfit(x, y, 1)


def crear_grafico(df, prod):
    df_prod = df[df['Producto'] == prod].copy()
    
    df_prod['Fecha'] = pd.to_datetime(df_prod['Año'].astype(str) + '-' + 
                                       df_prod['Mes'].astype(str), format='%Y-%m')
    
    ventas_mens = df_prod.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
    
    x = np.arange(len(ventas_mens))
    y = ventas_mens['Unidades_vendidas'].values
    
    pendiente, interseccion = calc_tendencia(x, y)
    tendencia = pendiente * x + interseccion
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ventas_mens['Fecha'], ventas_mens['Unidades_vendidas'], 
            label=prod, color='#5192be', linewidth=2)
    
    ax.plot(ventas_mens['Fecha'], tendencia, 
            label='Tendencia', color='red', linestyle='--', linewidth=2)
    
    ax.set_title('Evolución de Ventas Mensuales')
    ax.set_xlabel('Año-Mes')
    ax.set_ylabel('Unidades Vendidas')
    
    ax.set_ylim(bottom=0)
    
    ax.grid(True, which='major', axis='both', linestyle='-', alpha=0.5, linewidth=2.0) 
    ax.grid(True, which='minor', axis='x', linestyle='-', alpha=0.2, linewidth=0.5)
    
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_major_locator(YearLocator())
    
    ax.legend()
    
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig


def app():
    st.set_page_config(layout="wide")

    with st.sidebar:
        st.title("Subir Datos")
        archivo_csv = st.file_uploader("Selecciona CSV", type=['csv'])
        
        if archivo_csv is not None:
            df = cargar_datos(archivo_csv)
            sucursales = ['Todas'] + list(df['Sucursal'].unique())
            sucursal = st.selectbox('Sucursal', sucursales)
        else:
            sucursal = None
            df = None

    if df is None:
        st.header("Por favor, sube un CSV desde la barra lateral")
    
    if df is not None:
        if sucursal != 'Todas':
            df = df[df['Sucursal'] == sucursal]
        
        st.title("**Datos de Todas las Sucursales**" if sucursal == 'Todas' 
                else f"Datos de {sucursal}")
        
        for producto in df['Producto'].unique():
            with st.container(border=True):
                st.subheader(producto)
                
                (precio, var_precio, 
                margen, var_margen,
                unidades, var_unidades) = calc_metricas(df, producto)
                
                col_metricas, col_grafico = st.columns([1, 2], gap="large") 
                
                with col_metricas:
                    st.write("### Métricas")
                    st.metric("Precio Promedio", 
                            f"${precio:,.0f}", 
                            f"{var_precio:+.2f}%")
                    
                    st.metric("Margen Promedio",
                            f"{margen:.0f}%", 
                            f"{var_margen:+.2f}%")
                    
                    st.metric("Unidades Vendidas",
                            f"{unidades:,.0f}", 
                            f"{var_unidades:+.2f}%")
                
                with col_grafico:
                    fig = crear_grafico(df, producto)
                    fig.set_figwidth(10)
                    fig.set_figheight(6)
                    st.pyplot(fig)
                
                plt.close()

if __name__ == "__main__":
    app()
