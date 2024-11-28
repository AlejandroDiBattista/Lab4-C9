import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('Legajo: 59314')
        st.markdown('Nombre: Alejandro Yapura')
        st.markdown('Comisión: C9')

def cargar_datos(file):
    try:
        df = pd.read_csv(file)
        
        df['Año'] = df['Año'].astype(str)
        df['Mes'] = df['Mes'].astype(str).str.zfill(2)
        
        df['Fecha'] = pd.to_datetime(df['Año'] + '-' + df['Mes'] + '-01', format='%Y-%m-%d')
        
        return df
    except Exception as e:
        st.sidebar.error(f"Error al cargar el archivo: {str(e)}")
        return None

def calcular_metricas(df, producto, periodo_anterior, sucursal=None):
    if sucursal and sucursal != "Todas":
        df = df[df['Sucursal'] == sucursal]
    
    df_producto = df[df['Producto'] == producto]
    
    if len(df_producto) == 0:
        return 0, 0, 0, 0, 0, 0
    
    if df_producto[['Unidades_vendidas', 'Ingreso_total', 'Costo_total']].isnull().any().any():
        st.warning("Se encontraron valores nulos en las columnas necesarias.")
        return 0, 0, 0, 0, 0, 0
    
    unidades_vendidas = df_producto['Unidades_vendidas'].sum()
    ingreso_total = df_producto['Ingreso_total'].sum()
    costo_total = df_producto['Costo_total'].sum()
    
    if unidades_vendidas == 0:
        precio_promedio = 0
    else:
        precio_promedio = (df_producto['Ingreso_total'] / df_producto['Unidades_vendidas']).sum() / len(df_producto)
    
    if ingreso_total == 0:
        margen_promedio = 0
    else:
        margen_promedio = ((ingreso_total - costo_total) / ingreso_total) * 100
    
    df_anterior = df_producto[df_producto['Fecha'] < periodo_anterior]
    if len(df_anterior) > 0:
        ingreso_anterior = df_anterior['Ingreso_total'].sum()
        unidades_anterior = df_anterior['Unidades_vendidas'].sum()
        
        precio_anterior = ingreso_anterior / unidades_anterior if unidades_anterior != 0 else 0
        margen_anterior = ((ingreso_anterior - df_anterior['Costo_total'].sum()) / ingreso_anterior) * 100 if ingreso_anterior != 0 else 0
        
        var_precio = ((precio_promedio - precio_anterior) / precio_anterior) * 100 if precio_anterior != 0 else 0
        var_margen = margen_promedio - margen_anterior
        var_unidades = ((unidades_vendidas - unidades_anterior) / unidades_anterior) * 100 if unidades_anterior != 0 else 0
    else:
        var_precio = 0
        var_margen = 0
        var_unidades = 0
    
    return precio_promedio, margen_promedio, unidades_vendidas, var_precio, var_margen, var_unidades


def crear_grafico_tendencia(df, producto, sucursal=None):
    from matplotlib.dates import MonthLocator, YearLocator, DateFormatter
    
    if sucursal and sucursal != "Todas":
        df = df[df['Sucursal'] == sucursal]
    
    df_producto = df[df['Producto'] == producto]
    
    ventas_mensuales = df_producto.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
    
    if len(ventas_mensuales) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No hay datos disponibles', horizontalalignment='center')
        return fig
    
    x = np.arange(len(ventas_mensuales))
    z = np.polyfit(x, ventas_mensuales['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(ventas_mensuales['Fecha'], ventas_mensuales['Unidades_vendidas'], 
            label=producto, color='blue', linewidth=1)
    ax.plot(ventas_mensuales['Fecha'], p(x), '--', color='red', 
            label='Tendencia', linewidth=1)
    
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator())
    
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle='--', alpha=0.2)
    
    ax.set_xlabel('Año')
    ax.set_ylabel('Unidades Vendidas')
    ax.set_title('Evolución de Ventas Mensual')
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2, which='minor')
    ax.grid(True, alpha=0.6, which='major')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig

def mostrar_metricas(col, precio_promedio, margen_promedio, unidades_vendidas, 
                    var_precio, var_margen, var_unidades):
    col.metric(
        label="Precio Promedio", 
        value=f"${precio_promedio:,.0f}",
        delta=f"{var_precio:+.2f}%" if var_precio != 0 else None,
        delta_color="normal"
    )

    col.metric(
        label="Margen Promedio", 
        value=f"{margen_promedio:.1f}%",
        delta=f"{var_margen:+.2f}%" if var_margen != 0 else None,
        delta_color="inverse" if margen_promedio < 0 else "normal"
    )

    col.metric(
        label="Unidades Vendidas", 
        value=f"{unidades_vendidas:,.0f}",
        delta=f"{var_unidades:+.2f}%" if var_unidades != 0 else None,
        delta_color="inverse" if var_unidades < 0 else "normal"
    )

def main():
    st.set_page_config(layout="wide")
    
    with st.sidebar:
        st.header("Cargar archivo de datos")
        uploaded_file = st.file_uploader("Subir archivo CSV", type=['csv'])
    
    if uploaded_file is None:
        st.title("Por favor, sube un archivo CSV desde la barra lateral.")
        mostrar_informacion_alumno()
    else:
        df = cargar_datos(uploaded_file)
        
        if df is not None:
            with st.sidebar:
                sucursales = ["Todas"] + sorted(df['Sucursal'].unique().tolist())
                sucursal = st.selectbox("Seleccionar Sucursal", sucursales)
            
            ultima_fecha = df['Fecha'].max()
            periodo_anterior = ultima_fecha - pd.DateOffset(years=1)
            
            if sucursal == "Todas":
                st.title("Datos de Todas las Sucursales")
            else:
                st.title(f"Datos de {sucursal}")

            for producto in sorted(df['Producto'].unique()):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader(producto)
                        precio_promedio, margen_promedio, unidades_vendidas, \
                        var_precio, var_margen, var_unidades = calcular_metricas(
                            df, producto, periodo_anterior, sucursal)
                        
                        mostrar_metricas(st, precio_promedio, margen_promedio, unidades_vendidas,
                                       var_precio, var_margen, var_unidades)
                    
                    with col2:
                        fig = crear_grafico_tendencia(df, producto, sucursal)
                        st.pyplot(fig)
                st.write("")

if __name__ == "__main__":
    main()