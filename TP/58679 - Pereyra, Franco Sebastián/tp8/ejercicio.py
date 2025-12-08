import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
url = 'https://tp8-laboratorio-58679.streamlit.app/'


# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Ventas",
    layout="wide"
)

# Función para mostrar información básica del usuario
def mostrar_info_usuario():
    st.markdown("# Sube un archivo CSV para empezar")
    with st.container(border=True):
        st.markdown("**Legajo:** 58679")
        st.markdown("**Nombre:** Pereyra Franco") 
        st.markdown("**Comisión:** C9")

# Cargar y procesar el archivo subido
def cargar_datos():
    archivo_subido = st.sidebar.file_uploader("Cargar archivo CSV", type="csv")
    if archivo_subido:
        df = pd.read_csv(archivo_subido)
        df.columns = ['sucursal', 'producto', 'año', 'mes', 'unidades', 'ingreso', 'costo']
        df['orden'] = df.index  # Guardar el orden original
        return df, True
    else:
        return None, False

# Calcular métricas a partir de los datos
def obtener_metricas(df):
    df['precio_promedio'] = df['ingreso'] / df['unidades']
    df['margen'] = (df['ingreso'] - df['costo']) / df['ingreso']
    resumen = df.groupby('producto', sort=False).agg({
        'precio_promedio': 'mean',
        'margen': 'mean',
        'unidades': 'sum',
        'orden': 'min'
    }).reset_index()
    resumen = resumen.sort_values('orden')
    return resumen

# Crear gráfico de evolución de ventas
def crear_grafico(df, producto):
    agrupado = df.groupby(['año', 'mes']).agg({
        'unidades': 'sum'
    }).reset_index()
    agrupado['mes'] = agrupado['mes'].astype(str).str.zfill(2)
    agrupado['fecha'] = pd.to_datetime(agrupado['año'].astype(str) + '-' + agrupado['mes'] + '-01')
    agrupado = agrupado.sort_values('fecha')

    fig, ax = plt.subplots(figsize=(9, 6))  # Tamaño más amplio
    ax.plot(agrupado['fecha'], agrupado['unidades'], label=f"{producto}", color='#1E88E5', linewidth=2)
    z = np.polyfit(np.arange(len(agrupado)), agrupado['unidades'], 1)
    p = np.poly1d(z)
    ax.plot(agrupado['fecha'], p(np.arange(len(agrupado))), linestyle='--', color='crimson', label="Tendencia", linewidth=1.5)

    ax.set_title(f"Evolución de Ventas Mensual - {producto}", fontsize=12)
    ax.set_ylabel("Unidades Vendidas")
    ax.legend()
    return fig

# Formatear números con el estilo de la imagen
def formatear_numero(numero):
    return f"{numero:,.0f}"

def mostrar_metricas(producto, precio_promedio, margen, unidades, cambio_precio, cambio_unidades):
    st.metric("Precio Promedio", f"${formatear_numero(precio_promedio)}", f"{cambio_precio:.2f}%", delta_color="normal")
    st.metric("Margen Promedio", f"{margen:.0%}", f"{cambio_unidades:.2f}%", delta_color="normal")
    st.metric("Unidades Vendidas", formatear_numero(unidades), f"{cambio_unidades:.2f}%", delta_color="normal")

# Función principal
def main():
    # Cargar datos
    df, datos_cargados = cargar_datos()

    if not datos_cargados:
        mostrar_info_usuario()
        return

    st.title("Dashboard de Ventas")

    # Selección de sucursal
    sucursales = ["Todas"] + df['sucursal'].unique().tolist()
    sucursal_seleccionada = st.sidebar.selectbox("Elige una Sucursal", sucursales)
    if sucursal_seleccionada != "Todas":
        df = df[df['sucursal'] == sucursal_seleccionada]

    # Verificar si las métricas ya están almacenadas en el estado de la sesión
    if 'metricas' not in st.session_state:
        # Calcular métricas solo si no están almacenadas
        st.session_state.metricas = obtener_metricas(df)

    # Usar las métricas almacenadas
    metricas = st.session_state.metricas

    for _, fila in metricas.iterrows():
        st.divider()  # Línea divisoria entre productos
        with st.container(border=True):
            st.subheader(fila['producto'])

            # Crear columnas para valores (col1) y gráficos (col2)
            col1, col2 = st.columns([2, 3], gap="large")  # Proporción ajustada

            # Columna 1: Métricas
            with col1:
                cambio_precio = np.random.uniform(-20, 20)  # Simular cambios
                cambio_unidades = np.random.uniform(-10, 10)  # Simular cambios
                mostrar_metricas(
                    producto=fila['producto'],
                    precio_promedio=fila['precio_promedio'],
                    margen=fila['margen'],  
                    unidades=fila['unidades'],
                    cambio_precio=cambio_precio,
                    cambio_unidades=cambio_unidades
                )

            # Columna 2: Gráfico
            with col2:
                datos_producto = df[df['producto'] == fila['producto']]
                grafico = crear_grafico(datos_producto, fila['producto'])
                st.pyplot(grafico)

if __name__ == "__main__":
    main()
