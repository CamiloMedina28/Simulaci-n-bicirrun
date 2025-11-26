import streamlit as st
import folium
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components

ESTACIONES = {"CYT": (4.638181, -74.085202), 
              "Uriel": (4.638734, -74.088606),
              "Calle-26": (4.633039, -74.083574),
              "Calle-45": (4.635553, -74.080413),
              "Calle-53": (4.643666, -74.083148)}

INITIAL_BIKES = {
    "CYT": 33,
    "Uriel": 28,
    "Calle-26": 22,
    "Calle-45": 15,
    "Calle-53": 5
}

TOTAL_BICICLETAS_OPERATIVAS = 103

INITIAL_BIKES = {
    "CYT": 33,
    "Uriel": 28,
    "Calle-26": 22,
    "Calle-45": 15,
    "Calle-53": 5
}

user_bikes = {}


class SimulacionBicirrun():
    pass

def __asignar_color(cantidad):
        if cantidad > 20:
            return "green"
        elif cantidad >= 10:
            return "orange"
        else:
            return "red"

def __draw_map():
    m = folium.Map(location=[4.638, -74.084], zoom_start = 15)

    for estacion in ESTACIONES:
        cantidad = user_bikes[estacion]
        color = __asignar_color(cantidad)

        folium.Marker(
            location=[ESTACIONES[estacion][0], ESTACIONES[estacion][1]],
            popup=f"{estacion}: {cantidad} bicicletas disponibles",
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    return m._repr_html_()

st.title("Simulación Bicirrun")




for bike in INITIAL_BIKES:
    user_bikes[bike] = st.slider(f"Bicicletas en: {bike}",
                                0,
                                50,
                                INITIAL_BIKES[bike])

suma_total = sum(user_bikes.values())
st.write(f"{suma_total} bicicletas asignadas de {TOTAL_BICICLETAS_OPERATIVAS}")

if suma_total > TOTAL_BICICLETAS_OPERATIVAS:
    st.error(f"El límite máximo de bicicletas es de: {TOTAL_BICICLETAS_OPERATIVAS}, se han asignado {suma_total} lo cual supera el límite de la flota")
else:
    

    components.html(__draw_map(), height=500)

    df = pd.DataFrame({
        "Estación": list(user_bikes.keys()),
        "Bicicletas": list(user_bikes.values())
    })

    st.subheader("Gráfica 1: Disponibilidad por estación (Barras)")
    fig1 = px.bar(df, x="Estación", y="Bicicletas", text="Bicicletas")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Gráfica 2: Distribución porcentual de bicicletas (Pie Chart)")
    fig2 = px.pie(df, values="Bicicletas", names="Estación")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Gráfica 3: Evolución simulada de disponibilidad (placeholder)")
    timeline_df = pd.DataFrame({
        "Tiempo": list(range(1, 6)),
        "CYT": [user_bikes["CYT"] - i for i in range(5)],
        "Uriel": [user_bikes["Uriel"] - i for i in range(5)]
    })
    fig3 = px.line(timeline_df, x="Tiempo", y=["CYT", "Uriel"])
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Gráfica 4: Heatmap simple de disponibilidad")
    heat_df = df.set_index("Estación")
    fig4 = px.imshow([heat_df["Bicicletas"]], 
                    labels=dict(x="ESTACIONES", y="", color="Bicicletas"),
                    x=heat_df.index)
    st.plotly_chart(fig4, use_container_width=True)
