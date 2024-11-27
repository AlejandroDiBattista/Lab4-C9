import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ATENCION: Debe colocar la dirección en la que ha sido publicada la aplicación en la siguiente línea
url = 'https://tp8-saravia-zmvv3xptvq8dacbutwytcz.streamlit.app/'

# Función para mostrar la información del alumno
def mostrar_informacion_alumno():
    st.markdown("""
        <div style="border: 2px solid black; padding: 10px;">
            <p><strong>Legajo:</strong> 59.055</p>
            <p><strong>Nombre:</strong> Saravia Franco Nicolas</p>
            <p><strong>Comisión:</strong> C9</p>
        </div>
    """, unsafe_allow_html=True)

# Mostrar la información del alumno
mostrar_informacion_alumno()

# Cargar archivo CSV
st.sidebar.header("Seleccione un archivo CSV")
uploaded_file = st.sidebar.file_uploader("Seleccione un archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer archivo CSV
    df = pd.read_csv(uploaded_file)

    # Selección de sucursal en la barra lateral
    st.sidebar.subheader("Seleccionar Sucursal")
    sucursales = st.sidebar.selectbox("Seleccione una Sucursal", df['Sucursal'].unique())
    df_sucursal = df[df['Sucursal'] == sucursales]

    # Mostrar los datos de la sucursal seleccionada
    st.title(f"Datos de la {sucursales}")
    st.write(df_sucursal)

    # Calcular resultados por producto
    resultados = df.groupby("Producto").agg(
        Precio_Promedio=("Ingreso_total", lambda x: np.sum(x) / np.sum(df.loc[x.index, "Unidades_vendidas"])),
        Margen_Promedio=("Ingreso_total", lambda x: np.mean((x - df.loc[x.index, "Costo_total"]) / x) * 100),
        Unidades_Vendidas=("Unidades_vendidas", "sum"),
    ).reset_index()

    st.write("Resultados por Producto:")
    st.dataframe(resultados)

    # Mostrar gráfico por cada producto
    productos = resultados["Producto"].unique()
    for producto in productos:
        st.write(f"**{producto}**")
        producto_data = df[df["Producto"] == producto]

        # Ordenar por Año y Mes
        producto_data = producto_data.sort_values(by=["Año", "Mes"])

        # Crear un eje temporal basado en Año y Mes
        eje_temporal = producto_data["Año"].astype(str) + "-" + producto_data["Mes"].astype(str).str.zfill(2)

        # Crear gráfico de unidades vendidas
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            eje_temporal,
            producto_data["Unidades_vendidas"],
            label="Unidades Vendidas",
            color="blue",
        )
        ax.set_xticks(eje_temporal[::max(len(eje_temporal) // 10, 1)])  # Reducir etiquetas si hay muchas
        ax.set_xticklabels(eje_temporal[::max(len(eje_temporal) // 10, 1)], rotation=45)
        ax.set_xlabel("Período (Año-Mes)")
        ax.set_ylabel("Unidades Vendidas")

        # Línea de tendencia
        tendencia = np.polyfit(np.arange(len(producto_data)), producto_data["Unidades_vendidas"], 1)
        ax.plot(
            eje_temporal,
            np.polyval(tendencia, np.arange(len(producto_data))),
            label="Tendencia",
            color="red",
        )

        ax.legend()
        st.pyplot(fig)
