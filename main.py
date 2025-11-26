import streamlit as st
import folium
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components
from datetime import time
import numpy as np


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

class Graficar_demanda():
    def __init__(self, hora_inicio, hora_fin) -> None:
        self.hora_inicio = hora_inicio
        self.hora_fin = hora_fin

        self.horas_pico_cyt = [9, 11, 13, 14, 16]
        self.amplitud_horas_pico_cyt = [0.3, 0.3, 0.25, 0.25, 0.3]
        self.peso_horas_pico_cyt = [0.6, 0.6, 0.8, 0.7, 0.5]

        self.horas_pico_uriel = [6.75, 13, 16]
        self.amplitud_horas_pico_uriel = [0.2, 0.25, 0.3]
        self.peso_horas_pico_uriel = [1, 0.8, 0.5]

        self.horas_pico_45 = [6.75, 9, 11, 13, 14]
        self.amplitud_horas_pico_45 = [0.2, 0.15, 0.2, 0.18, 0.22]
        self.peso_horas_pico_45 = [0.8, 0.6, 0.5, 0.6, 0.6]
        
        self.horas_pico_26 = [6.75, 9, 11, 13, 14]
        self.amplitud_horas_pico_26 = [0.2, 0.3, 0.3, 0.25, 0.25]
        self.peso_horas_pico_26 = [0.7, 0.6, 0.5, 0.4, 0.6]

        self.horas_pico_53 = [9, 11, 12, 14, 16]
        self.amplitud_horas_pico_53 = [0.2, 0.25, 0.3, 0.25, 0.2]
        self.peso_horas_pico_53 = [0.3, 0.5, 0.4, 0.2, 0.3]

    def __demanda_gaussiana(self, t, mus, sigmas, amplitudes) -> int:
        """
        t: hora en formato decimal (ej: 7.5 para 7:30)
        mus: lista de horas pico
        sigmas: lista de anchos de pico
        amplitudes: lista de pesos (intensidad por pico)
        """
        t = np.array(t)
        total = 0
        for μ, σ, A in zip(mus, sigmas, amplitudes):
            total += A * np.exp(-((t - μ)**2) / (2 * σ**2))
        return total
    
    def __get_x_axis(self) -> list:
        return np.arange(self.hora_inicio, self.hora_fin, 0.015)

    
    def __get_data_frame(self, estacion:str):
        x = self.__get_x_axis()
        if estacion == "26":
            y = [self.__demanda_gaussiana(val, self.horas_pico_26, self.amplitud_horas_pico_26, self.peso_horas_pico_26) for val in x]
        elif estacion == "45":
            y = [self.__demanda_gaussiana(val, self.horas_pico_45, self.amplitud_horas_pico_45, self.peso_horas_pico_45) for val in x]
        elif estacion == "53":
            y = [self.__demanda_gaussiana(val, self.horas_pico_53, self.amplitud_horas_pico_53, self.peso_horas_pico_53) for val in x]
        elif estacion == "uriel":
            y = [self.__demanda_gaussiana(val, self.horas_pico_uriel, self.amplitud_horas_pico_uriel, self.peso_horas_pico_uriel) for val in x]
        else:
            y = [self.__demanda_gaussiana(val, self.horas_pico_cyt, self.amplitud_horas_pico_cyt, self.peso_horas_pico_cyt) for val in x]
        return x, y

    def get_grafica(self, estacion):
        x, y = self.__get_data_frame(estacion)
        df = pd.DataFrame({
            "Hora": x,
            "Demanda": y
        })
        return px.scatter(df, x="Hora", y = "Demanda")
    

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
        
        self.hora_inicio = st.time_input("Hora de inicio", value=time(6, 0))
        self.hora_fin = st.time_input("Hora de fin", value=time(16, 0))  
    
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
    
    def __graficar_demanda(self):
        st.subheader("Graficas de la demanda")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "CYT", "Uriel", "Calle 45", "Calle 53", "Calle 26"
        ])
        graficas = Graficar_demanda(float(self.hora_inicio.strftime("%H.%M")), float(self.hora_fin.strftime("%H.%M")))
        with tab1:
            st.plotly_chart(graficas.get_grafica("Cyt"), use_container_width=True, key = 5)
        with tab2:
            st.plotly_chart(graficas.get_grafica("uriel"), use_container_width=True, key = 6)
        with tab3:
            st.plotly_chart(graficas.get_grafica("45"), use_container_width=True, key = 7)
        with tab4:
            st.plotly_chart(graficas.get_grafica("53"), use_container_width=True, key = 8)
        with tab5:
            st.plotly_chart(graficas.get_grafica("26"), use_container_width=True, key = 9)
            

    def run(self):
        self.__generar_sliders()
        total = self.__sumar_bicicletas()

        if total > TOTAL_BICICLETAS_OPERATIVAS:
            st.error("Has excedido el límite máximo de bicicletas operativas.")
            return

        st.subheader("Mapa actualizado")
        components.html(self.__dibujar_mapa(), height=500)

        self.__graficar()
        self.__graficar_demanda()


app = SimulacionBicirrun()
app.run()
