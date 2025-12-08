import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## ATENCION Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://applab-xvugj6c8zegftvptacsfvw.streamlit.app/'

st.set_page_config(layout="wide", page_title="TP8-GASTON")

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('*Legajo:* 59056')
        st.markdown('*Nombre:* Villafañe Lucas Gaston')
        st.markdown('*Comisión:* C9')

def procesar_datos(df):
    df['Precio_promedio'] = df['Ingreso_total'] / df['Unidades_vendidas']
    df['Margen_promedio'] = (df['Ingreso_total'] - df['Costo_total']) / df['Ingreso_total']
    return df

def calcular_metricas_anteriores(df):
    metricas = ['Precio_promedio', 'Margen_promedio', 'Unidades_vendidas']
    for metrica in metricas:
        df[f'{metrica}_anterior'] = df.groupby('Producto')[metrica].shift(1)
    return df

def generar_resumen(df):
    return df.groupby('Producto').agg({
        'Precio_promedio': 'mean',
        'Precio_promedio_anterior': 'mean',
        'Margen_promedio': 'mean',
        'Margen_promedio_anterior': 'mean',
        'Unidades_vendidas': 'sum',
        'Unidades_vendidas_anterior': 'sum',
    }).reset_index()

def crear_grafico(datos_producto):
    fig, ax = plt.subplots(figsize=(26, 16))
    
    ax.plot(datos_producto['Fecha'], datos_producto['Unidades_vendidas'], label=datos_producto['Producto'].iloc[0], color="#2271b3", linestyle="-", linewidth=2)
    
    X = np.arange(len(datos_producto))
    y = datos_producto['Unidades_vendidas'].fillna(0).values
    coef = np.polyfit(X, y, 1)
    ax.plot(datos_producto['Fecha'], np.polyval(coef, X), label="Tendencia", linestyle="--", color="red", linewidth=1.5)
    
    ax.set_title("Evolución de Ventas Mensual", fontsize=28)
    ax.set_xlabel("Año", fontsize=24)
    ax.set_ylabel("Unidades Vendidas", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=20)
    
    plt.tight_layout()
    return fig

@st.cache_data
def cargar_datos(archivo):
    df = pd.read_csv(archivo)
    df['Mes'] = df['Mes'].astype(int)
    df['Año'] = df['Año'].astype(int)
    df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str) + '-01')
    return df

def main():
    st.sidebar.header("Cargar archivo de datos")
    archivo = st.sidebar.file_uploader("Subir archivo CSV", type=["csv"])

    if archivo is not None:
        df = cargar_datos(archivo)
        df = procesar_datos(df)
        df = calcular_metricas_anteriores(df)

        sucursales = ["Todas"] + list(df['Sucursal'].unique())
        sucursal = st.sidebar.selectbox("Seleccionar Sucursal", sucursales)

        if sucursal != "Todas":
            df = df[df['Sucursal'] == sucursal]
            st.header(f"Datos de la {sucursal}")
        else:
            st.header("Datos de todas las sucursales")

        resumen = generar_resumen(df)

        for _, row in resumen.iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"## {row['Producto']}")
                    for metrica in ['Precio_promedio', 'Margen_promedio', 'Unidades_vendidas']:
                        valor_actual = row[metrica]
                        valor_anterior = row[f'{metrica}_anterior']
                        delta = ((valor_actual - valor_anterior) / valor_anterior * 100) if valor_anterior else 0
                        
                        if metrica == 'Precio_promedio':
                            valor_formateado = f"${valor_actual:,.0f}".replace(",", ".")
                        elif metrica == 'Margen_promedio':
                            valor_formateado = f"{valor_actual * 100:.0f}%"
                        else:
                            valor_formateado = f"{int(valor_actual):,}"
                        
                        st.metric(
                            label=metrica.replace('_', ' ').title(),
                            value=valor_formateado,
                            delta=f"{delta:.2f}%"
                        )
                
                with col2:
                    datos_producto = df[df['Producto'] == row['Producto']].sort_values('Fecha')
                    fig = crear_grafico(datos_producto)
                    st.pyplot(fig)
    else:
        st.header("Por favor, sube un archivo CSV desde la barra lateral.")
        mostrar_informacion_alumno()

if __name__ == "__main__":
    main()