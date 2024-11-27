import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def obtencion_de_datos(ruta):
    tabla = pd.read_csv(ruta)
    tabla['PrecioMedio'] = tabla['Ingreso_total'] / tabla['Unidades_vendidas']
    tabla['MargenMedio'] = (tabla['Ingreso_total'] - tabla['Costo_total']) / tabla['Ingreso_total']
    tabla['Fecha'] = pd.to_datetime(tabla['Año'].astype(str) + '-' + tabla['Mes'].astype(str).str.zfill(2) + '-01')
    return tabla


def variaciones_anuales(datos_de_productos):
    agrupados = datos_de_productos.groupby('Año').agg({
        'PrecioMedio': 'mean',
        'MargenMedio': 'mean',
        'Unidades_vendidas': 'sum'
    }).reset_index()

    if len(agrupados) >= 2:
        ult, penult = agrupados.iloc[-1], agrupados.iloc[-2]
        delta_precio = ((ult['PrecioMedio'] / penult['PrecioMedio']) - 1) * 100
        delta_margen = ((ult['MargenMedio'] / penult['MargenMedio']) - 1) * 100
        delta_unidades = ((ult['Unidades_vendidas'] / penult['Unidades_vendidas']) - 1) * 100
        return delta_precio, delta_margen, delta_unidades
    return None, None, None


def generar_visual(datos_de_productos, etiqueta):
    agrupados = datos_de_productos.groupby(['Año', 'Mes']).agg({
        'Unidades_vendidas': 'sum',
        'Ingreso_total': 'sum'
    }).reset_index()

    agrupados['fecha'] = pd.to_datetime(
        agrupados['Año'].astype(str) + '-' + agrupados['Mes'].astype(str).str.zfill(2) + '-01'
    )
    agrupados = agrupados.sort_values('fecha')

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(True, linestyle='-', color='gray', alpha=0.3)

    fechas_ordinales = [fecha.toordinal() for fecha in agrupados['fecha']]
    z = np.polyfit(fechas_ordinales, agrupados['Unidades_vendidas'], 1)
    p = np.poly1d(z)
    tendencia = p(fechas_ordinales)

    ax.plot(agrupados['fecha'], agrupados['Unidades_vendidas'], color='#2E75B6', label=f"Producto\n{etiqueta}")
    ax.plot(agrupados['fecha'], tendencia, color='red', linestyle='--', label='Tendencia')

    ax.set_title("Evolución de Ventas Mensual")
    ax.set_xlabel("Año-Mes")
    ax.set_ylabel("Unidades Vendidas")
    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    return fig


def convertir_numero(valor, prefijo=""):
    if isinstance(valor, (int, float)):
        if prefijo == "$":
            return f"${valor:,.2f}".replace(",", ".")
        elif prefijo == "%":
            return f"{valor:.2f}%"
        return f"{valor:,.2f}".replace(",", ".")
    return str(valor)


def principal():
    st.set_page_config(layout="wide")
    with st.sidebar:
        st.title("Subir archivo CSV")
        ruta = st.file_uploader("Subir CSV", type=["csv"])

    if not ruta:
        st.subheader("Por favor, sube un archivo CSV desde la barra lateral.")
        with st.container(border=True):
            st.markdown("""
                **Legajo**: 59057 
                        
                **Nombre**: Rocha Lourdes Gabriela
                          
                **Comisión**: 9
            """)
    else:
        datos = obtencion_de_datos(ruta)
        opciones = ["Todas"] + list(datos['Sucursal'].unique())
        seleccion = st.sidebar.selectbox("Sucursal", opciones)

        if seleccion != "Todas":
            datos = datos[datos['Sucursal'] == seleccion]

        for prod in datos['Producto'].unique():
            producto_datos = datos[datos['Producto'] == prod]
            prom_precio = producto_datos['PrecioMedio'].mean()
            prom_margen = producto_datos['MargenMedio'].mean()
            total_unidades = producto_datos['Unidades_vendidas'].sum()
            var_precio, var_margen, var_unidades = variaciones_anuales(producto_datos)

            with st.container(border=True):
                st.subheader(prod)

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Precio Promedio", convertir_numero(prom_precio, "$"), f"{var_precio:+.2f}%")
                    st.metric("Margen Promedio", convertir_numero(prom_margen * 100, "%"), f"{var_margen:+.2f}%")
                    st.metric("Unidades Vendidas", convertir_numero(total_unidades), f"{var_unidades:+.2f}%")

                with col2:
                    grafico = generar_visual(producto_datos, prod)
                    if grafico:
                        st.pyplot(grafico)


if __name__ == "__main__":
    principal()
