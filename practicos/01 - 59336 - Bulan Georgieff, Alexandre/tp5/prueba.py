import streamlit as st
import pandas as pd
import numpy as np

st.title('¡Hola, Streamlit!')

st.write("Esta es mi primera aplicación con Streamlit.")

# Crear algunos datos
data = pd.DataFrame({
    'Números': np.arange(1, 11),
    'Cuadrados': np.arange(1, 11) ** 2
})

st.write("Aquí hay una tabla de datos:")
st.dataframe(data)

# Gráfico simple
st.line_chart(data.set_index('Números'))
