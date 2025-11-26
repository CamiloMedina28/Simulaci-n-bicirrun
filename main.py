import streamlit as st
import folium
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components
from datetime import time
import numpy as np
import random

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

# matriz_probabilidad_mañana = [
#     [0, 0, 0.1, 0.6, 0.3],
#     [0.05, 0, 0.2, 0.35, 0.4],
#     [0.25, 0.15, 0,	0, 0.6],
#     [0.3, 0.2, 0, 0, 0.5],
#     [0.3, 0.2, 0.25, 0.25,0],
# ]

# matriz_probabilidad_media_tarde = [
#     [0, 0.05, 0.25, 0.35, 0.35],
#     [0.12, 0, 0.3, 0.12, 0.46],
#     [0.32, 0.12, 0, 0.15, 0.41],
#     [0.35, 0.4, 0, 0, 0.25],
#     [0.35, 0.28, 0.27, 0.1, 0]
# ]

# matriz_probabilidad_tarde = [
#     [0, 0.22, 0.35, 0.15, 0.28],
#     [0.12, 0, 0.35, 0.42, 0.11],
#     [0.25, 0.15, 0, 0.28, 0.32],
#     [0.3, 0.12, 0.05, 0, 0.53],
#     [0.45, 0.225, 0.32, 0.005, 0]
# ]
# Mañana (6:00 - 9:00)
matriz_probabilidad_mañana = [
    [0, 0.25, 0.2, 0.35, 0.2],   # CYT
    [0.3, 0, 0.25, 0.25, 0.2],   # Uriel
    [0.2, 0.2, 0, 0.3, 0.3],     # Calle-26
    [0.25, 0.25, 0.25, 0, 0.25], # Calle-45
    [0.4, 0.3, 0.2, 0.1, 0]      # Calle-53
]

# Media tarde (9:00 - 13:00)
matriz_probabilidad_media_tarde = [
    [0, 0.2, 0.2, 0.35, 0.25],   # CYT
    [0.25, 0, 0.2, 0.25, 0.3],   # Uriel
    [0.25, 0.25, 0, 0.25, 0.25], # Calle-26
    [0.2, 0.3, 0.2, 0, 0.3],     # Calle-45
    [0.35, 0.3, 0.15, 0.2, 0]    # Calle-53
]

# Tarde (13:00 - 16:00)
matriz_probabilidad_tarde = [
    [0, 0.25, 0.2, 0.3, 0.25],   # CYT
    [0.2, 0, 0.25, 0.3, 0.25],   # Uriel
    [0.2, 0.2, 0, 0.3, 0.3],     # Calle-26
    [0.25, 0.25, 0.25, 0, 0.25], # Calle-45
    [0.35, 0.25, 0.2, 0.2, 0]    # Calle-53
]


TOTAL_BICICLETAS_OPERATIVAS = 103

class GraficarProcesosDiarios():
    def __init__(self, x, y, numero_inicial):
        self.horas = x
        self.distribucion_probabilistica_demanda = y
        self.inicial = numero_inicial
    
    def uso(self):
        init = self.inicial
        return [init * demanda for demanda in self.distribucion_probabilistica_demanda]
    

class GraficarDemanda():
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
        t = np.array(t)
        total = 0
        for μ, σ, A in zip(mus, sigmas, amplitudes):
            total += A * np.exp(-((t - μ)**2) / (2 * σ**2))
        return total
    
    def __get_x_axis(self) -> list:
        return np.arange(self.hora_inicio, self.hora_fin, 0.015)
    
    def __get_data_frame(self, estacion:str):
        x = self.__get_x_axis()
        if estacion == "Calle-26":
            y = [self.__demanda_gaussiana(val, self.horas_pico_26, self.amplitud_horas_pico_26, self.peso_horas_pico_26) for val in x]
        elif estacion == "Calle-45":
            y = [self.__demanda_gaussiana(val, self.horas_pico_45, self.amplitud_horas_pico_45, self.peso_horas_pico_45) for val in x]
        elif estacion == "Calle-53":
            y = [self.__demanda_gaussiana(val, self.horas_pico_53, self.amplitud_horas_pico_53, self.peso_horas_pico_53) for val in x]
        elif estacion == "Uriel":
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

    def get_data(self, estacion:str):
        return self.__get_data_frame(estacion)
    

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
        tab1, tab2, tab3 = st.tabs([
            "Barras", "Diagrama de torta", "Mapa de calor"
        ])

        with tab1:
            st.plotly_chart(self.__gen_bar_praph(df), use_container_width=True, key = 1)

        with tab2:
            st.plotly_chart(self.__gen_pie_chart(df), use_container_width=True, key = 2)
            
        with tab3:
            st.plotly_chart(self.__gen_heat_map(df), use_container_width=True, key = 4)
    
    def __graficar_demanda(self):
        st.subheader("Graficas de la demanda")
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "CYT", "Uriel", "Calle 45", "Calle 53", "Calle 26"
        ])
        graficas = GraficarDemanda(float(self.hora_inicio.strftime("%H.%M")), float(self.hora_fin.strftime("%H.%M")))
        with tab1:
            st.plotly_chart(graficas.get_grafica("CYT"), use_container_width=True, key = 5)
        with tab2:
            st.plotly_chart(graficas.get_grafica("Uriel"), use_container_width=True, key = 6)
        with tab3:
            st.plotly_chart(graficas.get_grafica("Calle-45"), use_container_width=True, key = 7)
        with tab4:
            st.plotly_chart(graficas.get_grafica("Calle-53"), use_container_width=True, key = 8)
        with tab5:
            st.plotly_chart(graficas.get_grafica("Calle-26"), use_container_width=True, key = 9)


    def __simular(self):
        data = GraficarDemanda(float(self.hora_inicio.strftime("%H.%M")), float(self.hora_fin.strftime("%H.%M")))
        demanda_por_estacion = {}
        for est in ESTACIONES:
            x, y = data.get_data(est)
            self.x = x
            demanda_por_estacion[est] = [y_i for y_i in y]
        
        self.historia = {est: [] for est in ESTACIONES}
        bicicletas = self.user_bikes.copy()
        
        # Lista para registrar los reabastecimientos
        self.reabastecimientos = []

        for hora_idx in range(len(x)):
            usuarios_salientes = {}
            usuarios_entrantes = {est: 0 for est in ESTACIONES}

            # 1️⃣ Calcular salidas
            for est in ESTACIONES:
                salida = int(bicicletas[est] * demanda_por_estacion[est][hora_idx])
                usuarios_salientes[est] = salida
                bicicletas[est] -= salida
            
            # 2️⃣ Distribuir a destinos según matriz de transición
            for i, est in enumerate(ESTACIONES):
                salida = usuarios_salientes[est]
                for _ in range(salida):
                    if x[hora_idx] <= 9:
                        destino = random.choices(list(ESTACIONES.keys()), weights=matriz_probabilidad_mañana[i], k=1)[0]
                    elif x[hora_idx] <= 13:
                        destino = random.choices(list(ESTACIONES.keys()), weights=matriz_probabilidad_media_tarde[i], k=1)[0]
                    else:
                        destino = random.choices(list(ESTACIONES.keys()), weights=matriz_probabilidad_tarde[i], k=1)[0]
                    usuarios_entrantes[destino] += 1
            
            # 3️⃣ Actualizar inventario con llegadas
            for est in ESTACIONES:
                bicicletas[est] += usuarios_entrantes[est]

            # 4️⃣ Revisar reabastecimientos
            for est in ESTACIONES:
                if bicicletas[est] <= 5:
                    # Encontrar estación con más bicicletas
                    donante = max(bicicletas, key=lambda k: bicicletas[k])
                    if donante != est and bicicletas[donante] > 5:
                        cantidad = min(5, bicicletas[donante]-5)  # ajustar cantidad a transferir
                        bicicletas[donante] -= cantidad
                        bicicletas[est] += cantidad
                        self.reabastecimientos.append({
                            "Hora": x[hora_idx],
                            "Estación_reabastecida": est,
                            "Estación_donante": donante,
                            "Cantidad": cantidad
                        })
            
            # 5️⃣ Guardar self.historia
            for est in ESTACIONES:
                self.historia[est].append(bicicletas[est])

        # Graficar evolución de bicicletas
        st.subheader("Comportamiento por horas del número de bicicletas")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(list(ESTACIONES.keys()))
        for idx, est in enumerate(ESTACIONES):
            df = pd.DataFrame({
                "Hora": x,
                "Bicicletas": self.historia[est]
            })
            fig = px.line(df, x="Hora", y="Bicicletas", title=f"Evolución de bicicletas en {est}")
            with [tab1, tab2, tab3, tab4, tab5][idx]:
                st.plotly_chart(fig, use_container_width=True)

        # Mostrar reabastecimientos
        if reabastecimientos:
            st.subheader("Reabastecimientos realizados")
            st.dataframe(pd.DataFrame(reabastecimientos))
        else:
            st.info("No se requirió ningún reabastecimiento durante la simulación.")

    def __graficar_todas_estaciones(self):
        st.subheader("Evolución de bicicletas en todas las estaciones")
        
        # Crear un DataFrame con todas las estaciones
        df = pd.DataFrame({"Hora": self.x})  # self.x = lista de horas de la simulación
        for est in ESTACIONES:
            df[est] = self.historia[est]  # historia[est] = lista de bicicletas por hora

        # Crear gráfico con varias líneas, cada estación con un color distinto
        fig = px.line(
            df,
            x="Hora",
            y=list(ESTACIONES.keys()),
            labels={"value": "Bicicletas", "variable": "Estación"},
            title="Número de bicicletas por estación",
            color_discrete_sequence=px.colors.qualitative.Dark24  # Paleta de colores
        )

        st.plotly_chart(fig, use_container_width=True)


    def __generate_maps(self):
        horas_clave = [7, 9, 11, 13, 14, 16]
        st.subheader("Mapas en horas clave con número de bicicletas")

        tabs = st.tabs([f"{h}h" for h in horas_clave])

        for idx, hora in enumerate(horas_clave):
            with tabs[idx]:
                m = folium.Map(location=[4.638, -74.084], zoom_start=15)
                for estacion in ESTACIONES:
                    # Encontrar el índice más cercano de la hora en tu simulación
                    hora_sim = np.array(self.x)  # x es la lista de horas de la simulación
                    hora_idx = (np.abs(hora_sim - hora)).argmin()
                    cantidad = self.historia[estacion][hora_idx]

                    color = self.__asignar_color(cantidad)
                    folium.Marker(
                        location=ESTACIONES[estacion],
                        popup=f"{estacion}: {cantidad} bicicletas",
                        icon=folium.Icon(color=color)
                    ).add_to(m)

                # st.components.v1.html(m._repr_html_(), height=500)
                components.html(m._repr_html_(), height=500)


            
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
        self.__simular()
        self.__graficar_todas_estaciones()
        self.__generate_maps()
        

app = SimulacionBicirrun()
app.run()
