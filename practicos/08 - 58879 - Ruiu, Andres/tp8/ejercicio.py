import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, YearLocator

url = 'https://tp8-58879.streamlit.app'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58879')
        st.markdown('**Nombre:** Andrés Ruiu')
        st.markdown('**Comisión:** C9')

def cargar_datos(archivo_csv):
    if archivo_csv is not None:
        df = pd.read_csv(archivo_csv)
        return df
    return None

def calcular_metricas(df, producto=None):
    df = df.copy()
    df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str), format='%Y-%m')
    
    df['Precio_unitario'] = df['Ingreso_total'].div(df['Unidades_vendidas'])
    df['Margen_promedio'] = ((df['Ingreso_total'] - df['Costo_total'])
                            .div(df['Ingreso_total'])
                            .fillna(0) * 100)
    
    df_producto = df[df['Producto'] == producto] if producto else df
    
    def calcular_metricas_periodo(data):
        return pd.DataFrame({
            'Precio_unitario': [data['Precio_unitario'].mean()],
            'Margen_promedio': [data['Margen_promedio'].mean()],
            'Unidades_vendidas': [data['Unidades_vendidas'].sum()]
        })
    
    metricas_generales = calcular_metricas_periodo(df_producto)
    
    años = sorted(df_producto['Año'].unique())
    
    if len(años) > 1:
        año_actual = años[-1]
        año_anterior = años[-2]
        
        metricas_año_actual = calcular_metricas_periodo(df_producto[df_producto['Año'] == año_actual])
        metricas_año_anterior = calcular_metricas_periodo(df_producto[df_producto['Año'] == año_anterior])
        
        variaciones = pd.DataFrame({
            'var_precio': [(metricas_año_actual['Precio_unitario'].iloc[0] / 
                           metricas_año_anterior['Precio_unitario'].iloc[0] - 1) * 100 
                          if metricas_año_anterior['Precio_unitario'].iloc[0] != 0 else 0],
            
            'var_margen': [metricas_año_actual['Margen_promedio'].iloc[0] - 
                          metricas_año_anterior['Margen_promedio'].iloc[0]],
            
            'var_unidades': [(metricas_año_actual['Unidades_vendidas'].iloc[0] / 
                             metricas_año_anterior['Unidades_vendidas'].iloc[0] - 1) * 100
                            if metricas_año_anterior['Unidades_vendidas'].iloc[0] != 0 else 0]
        })
    else:
        variaciones = pd.DataFrame({
            'var_precio': [0],
            'var_margen': [0],
            'var_unidades': [0]
        })
    
    return (
        metricas_generales['Precio_unitario'].iloc[0],
        variaciones['var_precio'].iloc[0],
        metricas_generales['Margen_promedio'].iloc[0],
        variaciones['var_margen'].iloc[0],
        metricas_generales['Unidades_vendidas'].iloc[0],
        variaciones['var_unidades'].iloc[0]
    )

def calcular_tendencia(x, y):
    return np.polyfit(x, y, 1)

def crear_grafico_ventas(df, producto):
    df_producto = df[df['Producto'] == producto].copy()
    
    df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + 
                                        df_producto['Mes'].astype(str), format='%Y-%m')
    
    ventas_mensuales = df_producto.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
    
    x = np.arange(len(ventas_mensuales))
    y = ventas_mensuales['Unidades_vendidas'].values
    
    slope, intercept = calcular_tendencia(x, y)
    tendencia = slope * x + intercept
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ventas_mensuales['Fecha'], ventas_mensuales['Unidades_vendidas'], 
            label=producto, color='#5192be', linewidth=2)
    
    ax.plot(ventas_mensuales['Fecha'], tendencia, 
            label='Tendencia', color='red', linestyle='--', linewidth=2)
    
    ax.set_title('Evolución de Ventas Mensual')
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

def main():
    st.set_page_config(layout="wide")

    with st.sidebar:
        st.title("Cargar archivo de datos")
        archivo_csv = st.file_uploader("Subir archivo CSV", type=['csv'])
        
        if archivo_csv is not None:
            df = cargar_datos(archivo_csv)
            sucursales = ['Todas'] + list(df['Sucursal'].unique())
            sucursal_seleccionada = st.selectbox('Seleccionar Sucursal', sucursales)
        else:
            sucursal_seleccionada = None
            df = None

    if df is None:
        st.header("Por favor, sube un archivo CSV desde la barra lateral")
        mostrar_informacion_alumno()
    
    if df is not None:
        if sucursal_seleccionada != 'Todas':
            df = df[df['Sucursal'] == sucursal_seleccionada]
        
        st.title("**Datos de Todas las Sucursales**" if sucursal_seleccionada == 'Todas' 
                else f"Datos de {sucursal_seleccionada}")
        
        for producto in df['Producto'].unique():
            with st.container(border=True):
                st.subheader(producto)
                
                (precio_promedio, var_precio, 
                margen_promedio, var_margen,
                unidades_vendidas, var_unidades) = calcular_metricas(df, producto)
                
                col_metricas, col_grafico = st.columns([1, 2], gap="large") 
                
                with col_metricas:
                    st.write("### Métricas")
                    st.metric("Precio Promedio", 
                            f"${precio_promedio:,.0f}", 
                            f"{var_precio:+.2f}%")
                    
                    st.metric("Margen Promedio",
                            f"{margen_promedio:.0f}%",
                            f"{var_margen:+.2f}%")
                    
                    st.metric("Unidades Vendidas",
                            f"{unidades_vendidas:,.0f}",
                            f"{var_unidades:+.2f}%")
                
                with col_grafico:
                    fig = crear_grafico_ventas(df, producto)
                    fig.set_figwidth(10)
                    fig.set_figheight(6)
                    st.pyplot(fig)
                
                plt.close()

if __name__ == "__main__":
    main()