
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Modelo poblacional completo", layout="centered")
st.title("📈 Modelos poblacionales: Exponencial, Logístico y Lineal")

# Explicación de tasa de crecimiento
with st.expander("📘 ¿Cómo se calcula la tasa de crecimiento?"):
    st.markdown(
        r"""
        La tasa de crecimiento porcentual muestra el cambio relativo de la población entre dos periodos.
        $$\text{Tasa}_{t}=\left(\frac{P_t}{P_{t-1}}-1\right)\times100$$
        """
    )

# Selección de modelo y presentación de fórmulas
st.markdown(
    r"""
    Selecciona un modelo:

    - **Exponencial**: $$P_{t+1}=P_t + r(1-s)P_t - m\frac{P_t}{100}$$
    - **Logístico**:  $$P_{t+1}=P_t + r(1-s)P_t\left(1-\tfrac{P_t}{K}\right) - m\frac{P_t}{100}$$
    - **Lineal**:      $$P_{t+1}=P_t + g(1-s)P_t - m\frac{P_t}{100}$$
    """
)

# Glosario de símbolos (antes de los widgets de entrada)
definitions = {
    "P_t": "Población total al periodo t",
    "H_t": "Población de hembras al periodo t",
    "r": "Tasa bruta de nacimientos por individuo (crías/animal/periodo)",
    "s": "Proporción de la población esterilizada (0–1)",
    "p": "Proporción inicial de hembras en la población total (0–1)",
    "m": "Tasa de mortalidad (muertes por cada 100 animales)",
    "K": "Capacidad de carga del ambiente",
    "g": "Tasa de crecimiento lineal por periodo (decimal, e.g., 0.10 para 10%)",
}
with st.expander("📘 Donde:"):
    for sym, desc in definitions.items():
        st.markdown(f"- **{sym}**: {desc}")

# Widgets de entrada
tab1, tab2 = st.columns(2)
with tab1:
    modelo = st.selectbox("Modelo", ["Exponencial", "Logístico", "Lineal"] )
    P0 = st.number_input("Población inicial total", 100, 50000, 1000, 100)
    pct_hembras = st.slider("Porcentaje inicial de hembras (%)", 0, 100, 40, 5)
    years = st.slider("Número de periodos", 1, 10, 3)
with tab2:
    r = st.number_input("Tasa de nacimientos (r)", 0.0, 20.0, 4.0, 0.1)
    s_pct = st.slider("Esterilización (%)", 0, 100, 35, 5)
    mortalidad = st.number_input("Mortalidad (por cada 100 animales)", 0.0, 100.0, 0.0, 0.1)
    K = None
    crecimiento_pct = 0.0
    if modelo == "Logístico":
        K = st.number_input("Capacidad de carga (K)", 1000, 100000, 10000, 500)
    elif modelo == "Lineal":
        crecimiento_pct = st.number_input("Crecimiento lineal (%) por periodo", 0.0, 500.0, 10.0, 0.1)

# Inicialización de variables
s = s_pct / 100
p = pct_hembras / 100
hembras = P0 * p
poblacion_total = P0

# Simulación
df_periods = np.arange(years + 1)
poblacion_general = [poblacion_total]
poblacion_solo_hembras = [hembras]

if modelo == "Lineal":
    g = crecimiento_pct / 100
    for _ in range(years):
        nuevas_gen = g * (1 - s) * poblacion_total
        nuevas_hem = g * (1 - s) * hembras
        poblacion_total += nuevas_gen - (mortalidad * poblacion_total / 100)
        hembras += nuevas_hem - (mortalidad * hembras / 100)
        poblacion_general.append(poblacion_total)
        poblacion_solo_hembras.append(hembras)
else:
    for _ in range(years):
        factor = 1 if modelo == "Exponencial" else (1 - poblacion_total / K)
        nuevas_crias_general = r * (1 - s) * poblacion_total * factor
        factor_h = 1 if modelo == "Exponencial" else (1 - (2 * hembras) / K)
        nuevas_crias_hembras = r * (1 - s) * hembras * p * factor_h
        poblacion_total += nuevas_crias_general - (mortalidad * poblacion_total / 100)
        hembras += nuevas_crias_hembras - (mortalidad * hembras / 100)
        poblacion_general.append(poblacion_total)
        poblacion_solo_hembras.append(hembras)

# Construcción del DataFrame
df = pd.DataFrame({
    "Periodo": df_periods,
    "Población General": poblacion_general,
    "Solo Hembras": poblacion_solo_hembras
})
# Tasas de crecimiento
df["Tasa creci. General (%)"] = df["Población General"].pct_change().fillna(0) * 100
df["Tasa creci. Hembras (%)"] = df["Solo Hembras"].pct_change().fillna(0) * 100
df = df.round(2)

# Gráfica con Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Periodo"], y=df["Población General"], mode='lines+markers', name='General'))
fig.add_trace(go.Scatter(x=df["Periodo"], y=df["Solo Hembras"], mode='lines+markers', name='Hembras'))
fig.update_layout(
    title="Crecimiento poblacional",
    xaxis_title="Periodo",
    yaxis_title="Número de animales",
    yaxis=dict(tickformat=",d"),
    template="simple_white"
)
st.plotly_chart(fig, use_container_width=True)

# Métricas finales
colA, colB, colC, colD = st.columns(4)

delta_gen = poblacion_general[-1] - P0
delta_hem = poblacion_solo_hembras[-1] - (P0 * p)
ester_tot = P0 * s
ester_hem = (P0 * p) * s
colA.metric("Población final General", f"{int(poblacion_general[-1]):,}", delta=f"{int(delta_gen):,}")
colB.metric("Población final Hembras", f"{int(poblacion_solo_hembras[-1]):,}", delta=f"{int(delta_hem):,}")
colC.metric("Total esterilizados", f"{int(ester_tot):,}", delta=f"{s_pct:.1f}%")
colD.metric("Hembras esterilizadas", f"{int(ester_hem):,}", delta=f"{100 * ester_hem/(P0*p):.1f}%")

# Tabla detallada
with st.expander("Ver tabla detallada"):
    st.dataframe(df)

st.caption("Comparación de tres modelos con ajuste de esterilización y mortalidad.")
