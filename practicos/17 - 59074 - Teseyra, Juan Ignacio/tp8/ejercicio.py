import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

# url= https://tp8-59074.streamlit.app/ #
def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('*Legajo:* 59074')
        st.markdown('*Nombre:* Teseyra, Juan Ignacio')
        st.markdown('*Comisión:* C9')

def cargar_datos(uploaded_file):
    """
    Carga y prepara los datos del archivo CSV.
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df.columns = ['Sucursal', 'Producto', 'Año', 'Mes', 'Unidades_vendidas', 'Ingreso_total', 'Costo_total']
            df['Periodo'] = df['Año'].astype(str) + "-" + df['Mes'].astype(str).str.zfill(2)
            # Guardamos el orden original de los productos
            productos_orden_original = df['Producto'].unique()
            # Solo ordenamos por Periodo manteniendo los productos en su orden original
            df = df.sort_values('Periodo')
            return df, productos_orden_original
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
    return None, None

def calcular_variaciones(df, productos_orden_original):
    """
    Calcula métricas y variaciones para cada producto.
    """
    resultados = []

    for producto in productos_orden_original:
        if producto in df['Producto'].unique():  # Solo procesamos si el producto existe en la sucursal
            grupo = df[df['Producto'] == producto].sort_values('Periodo')
            grupo['Precio_promedio'] = grupo['Ingreso_total'] / grupo['Unidades_vendidas']
            grupo['Margen_promedio'] = (grupo['Ingreso_total'] - grupo['Costo_total']) / grupo['Ingreso_total']

            if len(grupo) > 1:
                delta_precio = (grupo['Precio_promedio'].iloc[-1] - grupo['Precio_promedio'].iloc[-2]) / grupo['Precio_promedio'].iloc[-2] * 100
                delta_margen = (grupo['Margen_promedio'].iloc[-1] - grupo['Margen_promedio'].iloc[-2]) / grupo['Margen_promedio'].iloc[-2] * 100
                delta_unidades = (grupo['Unidades_vendidas'].iloc[-1] - grupo['Unidades_vendidas'].iloc[-2]) / grupo['Unidades_vendidas'].iloc[-2] * 100
            else:
                delta_precio = delta_margen = delta_unidades = None

            resultados.append({
                'Producto': producto,
                'Precio_promedio': grupo['Precio_promedio'].mean(),
                'Margen_promedio': grupo['Margen_promedio'].mean(),
                'Unidades_vendidas': grupo['Unidades_vendidas'].sum(),
                'Delta_precio': delta_precio,
                'Delta_margen': delta_margen,
                'Delta_unidades': delta_unidades
            })

    return pd.DataFrame(resultados)

def crear_grafico_ventas(df, producto):
    """
    Crea un gráfico de evolución de ventas para un producto.
    """
    ventas = df.groupby(['Año', 'Mes']).agg({'Ingreso_total': 'sum', 'Unidades_vendidas': 'sum'}).reset_index()
    ventas['Fecha'] = pd.to_datetime(ventas['Año'].astype(str) + '-' + ventas['Mes'].astype(str).str.zfill(2) + '-01')

    ventas.sort_values('Fecha', inplace=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(True, linestyle='-', alpha=0.3, color='gray')

    ax.plot(ventas['Fecha'], ventas['Unidades_vendidas'], color='#2563EB', linewidth=1.5, label=producto)
    
    z = np.polyfit(np.arange(len(ventas)), ventas['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    ax.plot(ventas['Fecha'], p(np.arange(len(ventas))), color='red', linestyle='--', linewidth=1.5, label='Tendencia')

    ax.set_title(f'Evolución de Ventas: {producto}', fontsize=12)
    ax.set_xlabel('Fecha', fontsize=10)
    ax.set_ylabel('Unidades Vendidas', fontsize=10)
    ax.legend(loc='best', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

    return fig

# Configuración de la app
uploaded_file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

if uploaded_file:
    df, productos_orden_original = cargar_datos(uploaded_file)
    if df is not None and productos_orden_original is not None:
        # Filtrar por sucursal
        sucursales = ['Todas'] + list(df['Sucursal'].unique())
        sucursal_seleccionada = st.sidebar.selectbox("Filtrar por Sucursal", sucursales)

        # Establecer el título según la sucursal seleccionada
        if sucursal_seleccionada == 'Todas':
            st.title("Análisis de ventas - Todas las sucursales")
        else:
            st.title(f"Análisis de ventas - Sucursal {sucursal_seleccionada}")

        # Aplicar el filtro en el DataFrame
        if sucursal_seleccionada != 'Todas':
            df_filtrado = df[df['Sucursal'] == sucursal_seleccionada]
        else:
            df_filtrado = df

        metricas = calcular_variaciones(df_filtrado, productos_orden_original)

        for producto in productos_orden_original:
            # Solo mostrar el producto si existe en la sucursal seleccionada
            if producto in df_filtrado['Producto'].unique():
                producto_data = metricas[metricas['Producto'] == producto].iloc[0]
                with st.container(border=True):
                    st.subheader(producto)

                    col1, col2 = st.columns([2, 3])

                    with col1:
                        st.metric("Precio Promedio", f"${producto_data['Precio_promedio']:,.2f}", f"{producto_data['Delta_precio']:.2f}%")
                        st.metric("Margen Promedio", f"{producto_data['Margen_promedio'] * 100:.2f}%", f"{producto_data['Delta_margen']:.2f}%")
                        st.metric("Unidades Vendidas", f"{producto_data['Unidades_vendidas']:,}", f"{producto_data['Delta_unidades']:.2f}%")

                    with col2:
                        datos_producto = df_filtrado[df_filtrado['Producto'] == producto]
                        grafico = crear_grafico_ventas(datos_producto, producto)
                        st.pyplot(grafico)
    else:
        st.error("No se pudo procesar el archivo.")
else:
    st.title("**Por favor sube un archivo CSV para comenzar**")
    mostrar_informacion_alumno()