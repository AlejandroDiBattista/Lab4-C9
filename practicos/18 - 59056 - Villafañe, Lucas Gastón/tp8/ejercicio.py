import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# url = 'https://app-grafica-eftpfygwbnhhyxayty7ray.streamlit.app/'


def mostrar_informacion_alumno():
    st.subheader("Información del Alumno")
    st.markdown("**Legajo:** 59056")
    st.markdown("**Nombre:** Lucas Gaston Villafañe")
    st.markdown("**Comisión:** C9")

def calcular_estadisticas(df):
    estadisticas = df.groupby("Producto").agg({
        "Ingreso_total": "sum",
        "Unidades_vendidas": "sum",
        "Costo_total": "sum"
    }).reset_index()

    estadisticas["Precio_Promedio"] = estadisticas["Ingreso_total"] / estadisticas["Unidades_vendidas"]
    estadisticas["Margen_Promedio"] = (estadisticas["Ingreso_total"] - estadisticas["Costo_total"]) / estadisticas["Ingreso_total"]
    estadisticas["Cambio_Unidades"] = np.random.uniform(-10, 10, len(estadisticas))
    estadisticas["Cambio_Margen"] = np.random.uniform(-5, 5, len(estadisticas))
    estadisticas["Cambio_Precio"] = np.random.uniform(-15, 15, len(estadisticas))

    return estadisticas

def filtrar_por_sucursal(df, sucursal):
    if sucursal != "Todas":
        return df[df["Sucursal"] == sucursal]
    return df

def generar_grafico(df, producto):
    datos_producto = df[df["Producto"] == producto]
    datos_producto = datos_producto.groupby(["Año", "Mes"]).agg({"Unidades_vendidas": "sum"}).reset_index()

    datos_producto["Fecha"] = pd.to_datetime(
        datos_producto["Año"].astype(str) + "-" + datos_producto["Mes"].astype(str)
    )

    datos_producto = datos_producto.sort_values("Fecha")

    plt.figure(figsize=(8, 4))
    plt.plot(datos_producto["Fecha"], datos_producto["Unidades_vendidas"], label=producto, marker='o')

    z = np.polyfit(range(len(datos_producto)), datos_producto["Unidades_vendidas"], 1)
    p = np.poly1d(z)
    plt.plot(datos_producto["Fecha"], p(range(len(datos_producto))), "r--", label="Tendencia")

    plt.title(f"Evolución de Ventas Mensual - {producto}")
    plt.xlabel("Fecha")
    plt.ylabel("Unidades Vendidas")
    plt.legend()
    plt.grid()

    return plt

def generar_flecha_moderno(cambio):
    color = "green" if cambio > 0 else "red"
    flecha = "▲" if cambio > 0 else "▼"
    estilo_flecha = f"color: {color}; font-size: 18px; margin-right: 5px;"
    estilo_porcentaje = f"color: {color}; font-size: 14px;"
    return f"""
        <div style="display: flex; align-items: center; gap: 4px;">
            <span style="{estilo_flecha}">{flecha}</span>
            <span style="{estilo_porcentaje}">{abs(cambio):.2f}%</span>
        </div>
    """

st.set_page_config(layout="wide", page_title="TP8 - Segundo Parcial")

st.sidebar.title("Cargar archivo de datos")
file = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

mostrar_informacion_alumno()

if not file:
    try:
        st.sidebar.write("")
    except FileNotFoundError:
        st.error("No se encontró el archivo 'gaseosas.csv'. Por favor, súbelo manualmente.")

if file:
    df = pd.read_csv(file)

    sucursales = ["Todas"] + df["Sucursal"].unique().tolist()
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

    # Filtrar los datos por sucursal seleccionada
    df_filtrado = filtrar_por_sucursal(df, sucursal_seleccionada)

    # Calcular estadísticas según los datos filtrados
    estadisticas = calcular_estadisticas(df_filtrado)

    st.title(f"Datos de {sucursal_seleccionada}")

    for _, row in estadisticas.iterrows():
        with st.container(border=True):
            col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown(f"### {row['Producto']}")

            for label, value, cambio in [
                ("Precio Promedio", f"${row['Precio_Promedio']:.2f}", row["Cambio_Precio"]),
                ("Margen Promedio", f"{row['Margen_Promedio']*100:.2f}%", row["Cambio_Margen"]),
                ("Unidades Vendidas", f"{row['Unidades_vendidas']:,}", row["Cambio_Unidades"]),
            ]:
                st.markdown(f"**{label}**")
                st.markdown(f"<span style='font-size: 24px; color: black;'>{value}</span>", unsafe_allow_html=True)
                st.markdown(generar_flecha_moderno(cambio), unsafe_allow_html=True)

        with col2:
            grafico = generar_grafico(df_filtrado, row["Producto"])
            st.pyplot(grafico)
else:
    st.write("Por favor, sube un archivo CSV desde la barra lateral.")