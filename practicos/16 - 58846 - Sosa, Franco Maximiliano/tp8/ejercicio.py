import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# url = 'https://tp8-58846.streamlit.app/'

st.set_page_config(layout="wide", page_title="Panel de Ventas")

def formatear_numero(valor):
    return f"{valor:,.0f}".replace(",", ".")

def calcular_metricas(datos):
    
    datos['Precio_promedio'] = datos['Ingreso_total'] / datos['Unidades_vendidas']
    precio_promedio = datos['Precio_promedio'].mean()
    precio_promedio_anual = datos.groupby('Año')['Precio_promedio'].mean()
    variacion_precio_promedio_anual = precio_promedio_anual.pct_change().mean() * 100

    
    datos['Ganancia'] = datos['Ingreso_total'] - datos['Costo_total']
    datos['Margen'] = (datos['Ganancia'] / datos['Ingreso_total']) * 100
    margen_promedio = datos['Margen'].mean()
    margen_promedio_anual = datos.groupby('Año')['Margen'].mean()
    variacion_margen_promedio_anual = margen_promedio_anual.pct_change().mean() * 100

    
    unidades_vendidas = datos['Unidades_vendidas'].sum()
    unidades_por_año = datos.groupby('Año')['Unidades_vendidas'].sum()
    variacion_anual_unidades = unidades_por_año.pct_change().mean() * 100

    
    datos_agrupados = datos.groupby(['Año', 'Mes']).agg({
        'Ingreso_total': 'sum',
        'Costo_total': 'sum',
        'Unidades_vendidas': 'sum'
    }).reset_index()
    
    datos_agrupados = datos_agrupados.sort_values(['Año', 'Mes'])
    
    if len(datos_agrupados) > 1:
        ultimo_periodo = datos_agrupados.iloc[-1]
        penultimo_periodo = datos_agrupados.iloc[-2]
        
        precio_ultimo = ultimo_periodo['Ingreso_total'] / ultimo_periodo['Unidades_vendidas']
        precio_penultimo = penultimo_periodo['Ingreso_total'] / penultimo_periodo['Unidades_vendidas']
        var_precio = ((precio_ultimo - precio_penultimo) / precio_penultimo) * 100
        
        margen_ultimo = ((ultimo_periodo['Ingreso_total'] - ultimo_periodo['Costo_total']) / ultimo_periodo['Ingreso_total']) * 100
        margen_penultimo = ((penultimo_periodo['Ingreso_total'] - penultimo_periodo['Costo_total']) / penultimo_periodo['Ingreso_total']) * 100
        var_margen = margen_ultimo - margen_penultimo
        
        var_unidades = ((ultimo_periodo['Unidades_vendidas'] - penultimo_periodo['Unidades_vendidas']) / penultimo_periodo['Unidades_vendidas']) * 100
    else:
        var_precio = 0
        var_margen = 0
        var_unidades = 0
    
    return (
        precio_promedio, 
        margen_promedio, 
        unidades_vendidas, 
        var_precio, 
        var_margen, 
        var_unidades,
        variacion_precio_promedio_anual,
        variacion_margen_promedio_anual,
        variacion_anual_unidades
    )

def crear_grafico_series_temporales(datos, producto):
    fig, ax = plt.subplots(figsize=(8,6))
    
    ventas = datos.groupby(['Año', 'Mes']).agg({
        'Ingreso_total': 'sum',
        'Unidades_vendidas': 'sum'
    }).reset_index()
    
    ventas['Fecha'] = pd.to_datetime(ventas['Año'].astype(str) + '-' + 
                                   ventas['Mes'].astype(str).str.zfill(2) + '-01')
    ventas = ventas.sort_values('Fecha')
    
    margen_superior = ventas['Unidades_vendidas'].max() * 1.0
    margen_inferior = max(0, ventas['Unidades_vendidas'].min() * 0.1)
    ax.set_ylim([margen_inferior, margen_superior])
    
    ax.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    ax.plot(ventas['Fecha'], ventas['Unidades_vendidas'], 
            color='blue', linewidth=1, marker='o', markersize=3,
            label=producto)
    
    z = np.polyfit(np.arange(len(ventas)), ventas['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(ventas['Fecha'], p(np.arange(len(ventas))), 
            color='red', linestyle='--', linewidth=1,
            label='Tendencia')
    
    ax.legend(loc='best', frameon=True, fontsize=8)
    ax.set_title('Evolución de Ventas Mensual', fontsize=10, pad=10)
    ax.set_ylabel('Unidades Vendidas', fontsize=8)
    ax.tick_params(axis='both', labelsize=14)
    plt.tight_layout()
    
    return fig

with st.sidebar:
    st.header("Cargar archivo de datos")
    archivo_cargado = st.file_uploader("", type=["csv"])

if archivo_cargado:
    datos = pd.read_csv(archivo_cargado)
    
    with st.sidebar:
        sucursal = st.selectbox(
            "Seleccionar Sucursal",
            options=["Todas", "Sucursal Norte", "Sucursal Centro", "Sucursal Sur"]
        )
    
    if sucursal != "Todas":
        datos = datos[datos["Sucursal"] == sucursal]
    
    st.title(f"Datos de {'Todas las Sucursales' if sucursal == 'Todas' else sucursal}")
    
    for producto in datos["Producto"].unique():
        datos_producto = datos[datos["Producto"] == producto]
        
        with st.container(border=True):
            col1, col2 = st.columns([1,2])
            
            with col1:
                st.subheader(producto)
                precio_prom, margen_prom, unidades, var_precio, var_margen, var_unidades, var_precio_anual, var_margen_anual, var_unidades_anual = calcular_metricas(datos_producto)
                
                st.metric(
                    label="Precio Promedio",
                    value=f"${formatear_numero(precio_prom)}",
                    delta=f"{var_precio_anual:+.2f}%"
                )
                
                st.metric(
                    label="Margen Promedio",
                    value=f"{margen_prom:.0f}%",
                    delta=f"{var_margen_anual:+.2f}%"
                )
                
                st.metric(
                    label="Unidades Vendidas",
                    value=formatear_numero(unidades),
                    delta=f"{var_unidades_anual:+.2f}%"
                )
            
            with col2:
                st.pyplot(crear_grafico_series_temporales(datos_producto, producto))

else:
    st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
    
    with st.container(border=True):
        st.markdown("""
            Legajo: 58.846

            Nombre: Sosa Franco Maximiliano

            Comisión: C9
        """)