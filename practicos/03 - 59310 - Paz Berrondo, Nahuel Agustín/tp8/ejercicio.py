import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-59310.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 59.310')
        st.markdown('**Nombre:** Paz Berrondo Nahuel Agustin')
        st.markdown('**Comisión:** C9')

# Procesar archivo subido
def procesar_archivo_subido() -> (pd.DataFrame, bool):
    archivo = st.sidebar.file_uploader(
        label="Subir archivo CSV", 
        type=["csv"]
    )
    if not archivo:
        return None, False
            
    df = pd.read_csv(archivo)
    df.columns = ['sucursal', 'producto', 'año', 'mes', 'unidades', 'ingresos', 'costo']
    df['secuencia_original'] = range(len(df))
    df['periodo'] = df['año'].astype(str) + "-" + df['mes'].astype(str).str.zfill(2)
    df = df.sort_values('periodo')
    return df, True

# Calcular métricas de negocio
def calcular_métricas_de_negocio(datos: pd.DataFrame) -> pd.DataFrame:
    analiticas = datos.copy()
    analiticas['precio_unitario'] = analiticas['ingresos'] / analiticas['unidades']
    analiticas['margen_de_ganancia'] = (analiticas['ingresos'] - analiticas['costo']) / analiticas['ingresos']
    
    return (analiticas.groupby('producto', as_index=False)
            .agg({
                'precio_unitario': 'mean',
                'margen_de_ganancia': 'mean',
                'unidades': 'sum',
                'secuencia_original': 'min'
            })
            .sort_values('secuencia_original'))

# Generar gráfico de tendencia de ventas
def generar_tendencia_de_ventas(datos: pd.DataFrame, nombre_producto: str) -> plt.Figure:
    try:
        datos_mensuales = (datos.groupby(['año', 'mes'])
                           .agg({
                               'ingresos': 'sum',
                               'unidades': 'sum'
                           })
                           .reset_index())

        datos_mensuales['fecha'] = pd.to_datetime(
            datos_mensuales['año'].astype(str) + '-' + 
            datos_mensuales['mes'].astype(str).str.zfill(2) + '-01'
        )
        datos_mensuales = datos_mensuales.sort_values('fecha')

        fig, ax = plt.subplots(figsize=(9, 5))
        
        y_max = datos_mensuales['unidades'].max() * 1.1
        y_min = max(0, datos_mensuales['unidades'].min() * 0.1)
        
        ax.set_ylim([y_min, y_max])
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        ax.grid(True, linestyle='-', alpha=0.3, color='gray')

        ax.plot(
            datos_mensuales['fecha'], 
            datos_mensuales['unidades'], 
            color='#FFFFFF', 
            label="Producto"
        )

        ax.plot(
            datos_mensuales['fecha'], 
            datos_mensuales['unidades'], 
            color='#2563EB', 
            linewidth=1.5, 
            markersize=4, 
            label=nombre_producto
        )

        # Calcular tendencia
        coeficientes = np.polyfit(
            np.arange(len(datos_mensuales)), 
            datos_mensuales['unidades'], 
            1
        )
        tendencia = np.poly1d(coeficientes)
        
        ax.plot(
            datos_mensuales['fecha'], 
            tendencia(np.arange(len(datos_mensuales))),
            color='red', 
            linestyle='--', 
            linewidth=1.5, 
            label='Tendencia'
        )

        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.set_title('Evolución de Ventas Mensual', fontsize=15, pad=10)
        ax.set_xlabel('Año-Mes', fontsize=10)
        ax.set_ylabel('Unidades vendidas', fontsize=10)
        ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        return fig
        
    except Exception as e:
        print(f"Error al visualizar para '{nombre_producto}': {e}")
        return None

# Calcular variaciones
def calcular_variaciones(df, productos_orden_original):
    datos_métricas = calcular_métricas_de_negocio(df)
    
    resultados = []
    for producto in productos_orden_original:
        if producto in df['producto'].unique():
            grupo = df[df['producto'] == producto].sort_values('periodo')
            
            # Calcular variaciones
            if len(grupo) > 1:
                precios = grupo['ingresos'] / grupo['unidades']
                precio_promedio = precios.mean()
                if len(precios) > 1:
                    delta_precio = ((precios.iloc[-1] - precio_promedio) / precio_promedio) * 100
                else:
                    delta_precio = None

                margenes = (grupo['ingresos'] - grupo['costo']) / grupo['ingresos']
                margen_promedio = margenes.mean()
                if len(margenes) > 1:
                    delta_margen = ((margenes.iloc[-1] - margen_promedio) / margen_promedio) * 100
                else:
                    delta_margen = None

                unidades = grupo['unidades']
                unidades_promedio = unidades.mean()
                if len(unidades) > 1:
                    delta_unidades = ((unidades.iloc[-1] - unidades_promedio) / unidades_promedio) * 100
                else:
                    delta_unidades = None

            else:
                delta_precio = delta_margen = delta_unidades = None
                precio_promedio = grupo['ingresos'].iloc[0] / grupo['unidades'].iloc[0]
                margen_promedio = (grupo['ingresos'].iloc[0] - grupo['costo'].iloc[0]) / grupo['ingresos'].iloc[0]
                unidades_promedio = grupo['unidades'].iloc[0]

            producto_data = datos_métricas[datos_métricas['producto'] == producto].iloc[0]
            resultados.append({
                'Producto': producto,
                'Precio_promedio': precio_promedio,
                'Margen_promedio': margen_promedio,
                'Unidades_vendidas': producto_data['unidades'],
                'Delta_precio': delta_precio,
                'Delta_margen': delta_margen,
                'Delta_unidades': delta_unidades
            })
    
    return pd.DataFrame(resultados)

# Formatear métricas para la visualización
def formatear_metrica(valor: float, tipo_formato: str) -> str:
    if valor is None:
        return "N/A"
    
    if tipo_formato == 'precio':
        return f"${valor:,.0f}".replace(",", ".")
    elif tipo_formato == 'porcentaje':
        return f"{valor:.2f}".replace(".", ",")
    elif tipo_formato == 'unidades':
        return f"{int(valor):,}".replace(",", ".")
    return str(valor)



# Renderizar el panel de la aplicación
def renderizar_panel():
    datos, cargado = procesar_archivo_subido()
    
    if not cargado:
        st.title("Por favor, sube un archivo CSV desde la barra lateral.")
        mostrar_informacion_alumno()
        return
    
    sucursales_disponibles = ["Todas"] + sorted(datos['sucursal'].unique().tolist())
    sucursal_seleccionada = st.sidebar.selectbox("Seleccionar Sucursal", sucursales_disponibles)
    
    if sucursal_seleccionada == "Todas":
        st.title("Datos de Todas las Sucursales")
    else:
        st.title(f"Datos de la {sucursal_seleccionada}")
    
    datos_filtrados = datos[datos['sucursal'] == sucursal_seleccionada] if sucursal_seleccionada != "Todas" else datos
    datos_métricas = calcular_métricas_de_negocio(datos_filtrados)
    
    metricas = calcular_variaciones(datos_filtrados, datos_métricas['producto'].tolist())
    
    for _, datos_producto in datos_métricas.iterrows():
        with st.container(border=True):
            st.subheader(datos_producto['producto'])
            columna_métricas, columna_grafico = st.columns([2, 4])
            
            deltas = metricas[metricas['Producto'] == datos_producto['producto']].iloc[0]
            
            with columna_métricas:
                st.metric(
                    "Precio Promedio",
                    formatear_metrica(datos_producto['precio_unitario'], 'precio'),
                    formatear_metrica(deltas['Delta_precio'], 'porcentaje') + "%" if deltas['Delta_precio'] is not None else "N/A"
                )
                st.metric(
                    "Margen Promedio",
                    formatear_metrica(datos_producto['margen_de_ganancia'] * 100, 'porcentaje') + "%",
                    formatear_metrica(deltas['Delta_margen'], 'porcentaje') + "%" if deltas['Delta_margen'] is not None else "N/A"
                )
                st.metric(
                    "Total Unidades",
                    formatear_metrica(datos_producto['unidades'], 'unidades'),
                    formatear_metrica(deltas['Delta_unidades'], 'porcentaje') + "%" if deltas['Delta_unidades'] is not None else "N/A"
                )
            
            with columna_grafico:
                subset_producto = datos_filtrados[datos_filtrados['producto'] == datos_producto['producto']]
                if grafico := generar_tendencia_de_ventas(subset_producto, datos_producto['producto']):
                    st.pyplot(grafico)

# Inicializar la aplicación
def inicializar_aplicacion():
    renderizar_panel()

if __name__ == "__main__":
    inicializar_aplicacion()