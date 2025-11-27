import streamlit as st
import folium
import streamlit.components.v1 as components
import plotly.express as px
import pandas as pd

st.title("Simulación Bicirrun")

# Coordenadas de las estaciones
estaciones = {
    "CYT": (4.638181, -74.085202),
    "Uriel": (4.638734, -74.088606),
    "Calle-26": (4.633039, -74.083574),
    "Calle-45": (4.635553, -74.080413),
    "Calle-53": (4.643666, -74.083148)
}

# Valores iniciales
initial_bikes = {
    "CYT": 33,
    "Uriel": 28,
    "Calle-26": 22,
    "Calle-45": 15,
    "Calle-53": 5
}

TOTAL = 103
user_bikes = {}

# --- SLIDERS EN DOS COLUMNAS ---
st.subheader("Inventario inicial por estación:")

col1, col2 = st.columns(2)

for i, station in enumerate(initial_bikes):
    if i % 2 == 0:
        user_bikes[station] = col1.slider(
            f"Bicicletas en {station}", 0, 50, initial_bikes[station]
        )
    else:
        user_bikes[station] = col2.slider(
            f"Bicicletas en {station}", 0, 50, initial_bikes[station]
        )

# Validar suma
suma = sum(user_bikes.values())
st.write(f"Total asignado: {suma}/{TOTAL}")

if suma > TOTAL:
    st.error("Supera el máximo permitido (103). Ajusta los valores.")
else:
    # --- MAPA ---
    with st.container():
        st.subheader("Mapa de disponibilidad")
        m = folium.Map(location=[4.638, -74.084], zoom_start=16)

        def color(c):
            if c > 20: return "green"
            if c >= 10: return "orange"
            return "red"

        for stn in estaciones:
            folium.Marker(
                location=estaciones[stn],
                popup=f"{stn}: {user_bikes[stn]} bicis",
                icon=folium.Icon(color=color(user_bikes[stn]))
            ).add_to(m)

        components.html(m._repr_html_(), height=500)

    # --- GRAFICAS ---
    df = pd.DataFrame({
        "Estación": user_bikes.keys(),
        "Bicicletas": user_bikes.values()
    })

    fig1 = px.bar(df, x="Estación", y="Bicicletas")
    fig2 = px.pie(df, values="Bicicletas", names="Estación")
    fig4 = px.imshow([df["Bicicletas"]], 
                     labels={"x": "Estaciones", "color": "Bicicletas"},
                     x=df["Estación"])

    # TABS PARA ORGANIZAR MEJOR
    st.subheader("Análisis gráfico")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Disponibilidad por estación",
        "Distribución porcentual",
        "Evolución temporal",
        "Heatmap"
    ])

    with tab1:
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.plotly_chart(fig4, use_container_width=True)
