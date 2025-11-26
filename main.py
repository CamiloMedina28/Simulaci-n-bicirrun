import streamlit as st
import folium
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components


ESTACIONES = {
    "CYT": (4.638181, -74.085202), 
    "Uriel": (4.638734, -74.088606),
    "Calle-26": (4.633039, -74.083574),
    "Calle-45": (4.635553, -74.080413),
    "Calle-53": (4.643666, -74.083148)
}

INITIAL_BIKES = {
    "CYT": 33,
    "Uriel": 28,
    "Calle-26": 22,
    "Calle-45": 15,
    "Calle-53": 5
}

TOTAL_BICICLETAS_OPERATIVAS = 103

class SimulacionBicirrun:
    def __init__(self):
        st.title("Simulación Bicirrun")
        self.user_bikes = {}

    def __generar_sliders(self):
        st.subheader("Inventario inicial por estación")

        for estacion in INITIAL_BIKES:
            self.user_bikes[estacion] = st.slider(
                f"Bicicletas en {estacion}",
                0, 50,
                INITIAL_BIKES[estacion]
            )
    
    def __sumar_bicicletas(self):
        total = sum(self.user_bikes.values())
        st.write(f"{total} bicicletas asignadas de {TOTAL_BICICLETAS_OPERATIVAS}")
        return total
    
    def __asignar_color(self, cantidad):
        if cantidad > 20:
            return "green"
        elif cantidad >= 10:
            return "orange"
        else:
            return "red"

    def __dibujar_mapa(self):
        m = folium.Map(location=[4.638, -74.084], zoom_start=15)

        for estacion in ESTACIONES:
            cantidad = self.user_bikes[estacion]
            color = self.__asignar_color(cantidad)

            folium.Marker(
                location=ESTACIONES[estacion],
                popup=f"{estacion}: {cantidad} bicicletas",
                icon=folium.Icon(color=color)
            ).add_to(m)

        return m._repr_html_()

    def __gen_bar_praph(self, df):
        fig1 = px.bar(df, x="Estación", y="Bicicletas", text="Bicicletas")
        return fig1
    
    def __gen_pie_chart(self, df):
        fig2 = px.pie(df, values="Bicicletas", names="Estación")
        return fig2

    def __gen_heat_map(self, df):
        heat_df = df.set_index("Estación")
        fig4 = px.imshow(
            [heat_df["Bicicletas"]],
            labels=dict(x="Estación", color="Bicicletas"),
            x=heat_df.index
        )
        return fig4


    def __graficar(self):
        df = pd.DataFrame({
            "Estación": list(self.user_bikes.keys()),
            "Bicicletas": list(self.user_bikes.values())
        })

        st.subheader("Gráficas de análisis")
        tab1, tab2, tab3, tab4 = st.tabs([
            "Barras", "Pie Chart", "Evolución", "Heatmap"
        ])

        with tab1:
            st.plotly_chart(self.__gen_bar_praph(df), use_container_width=True, key = 1)

        with tab2:
            st.plotly_chart(self.__gen_pie_chart(df), use_container_width=True, key = 2)

        with tab3:
            st.write("PENDIENTE")
            pass
            
        with tab4:
            st.plotly_chart(self.__gen_heat_map(df), use_container_width=True, key = 4)

    def run(self):
        self.__generar_sliders()
        total = self.__sumar_bicicletas()

        if total > TOTAL_BICICLETAS_OPERATIVAS:
            st.error("Has excedido el límite máximo de bicicletas operativas.")
            return

        st.subheader("Mapa actualizado")
        components.html(self.__dibujar_mapa(), height=500)

        self.__graficar()


app = SimulacionBicirrun()
app.run()
