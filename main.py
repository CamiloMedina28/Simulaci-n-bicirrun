import streamlit as st
import folium
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components
from datetime import time
import numpy as np
import random
import copy

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

CANTIDAD_TRANSFERIR = 15
MINIMO_POR_ESTACION = 3

# Mañana (6:00 - 9:00)
matriz_probabilidad_mañana = [
    [0, 0.25, 0.2, 0.35, 0.2],   
    [0.3, 0, 0.25, 0.25, 0.2],   
    [0.2, 0.2, 0, 0.3, 0.3],     
    [0.25, 0.25, 0.25, 0, 0.25], 
    [0.4, 0.3, 0.2, 0.1, 0]      
]

# Media tarde (9:00 - 13:00)
matriz_probabilidad_media_tarde = [
    [0, 0.2, 0.2, 0.35, 0.25],
    [0.25, 0, 0.2, 0.25, 0.3],   
    [0.25, 0.25, 0, 0.25, 0.25],
    [0.2, 0.3, 0.2, 0, 0.3],    
    [0.35, 0.3, 0.15, 0.2, 0]   
]

# Tarde (13:00 - 16:00)
matriz_probabilidad_tarde = [
    [0, 0.25, 0.2, 0.3, 0.25],   
    [0.2, 0, 0.25, 0.3, 0.25],   
    [0.2, 0.2, 0, 0.3, 0.3],     
    [0.25, 0.25, 0.25, 0, 0.25], 
    [0.35, 0.25, 0.2, 0.2, 0]    
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
        return np.arange(self.hora_inicio, self.hora_fin, 0.05)
    
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
        self.horas = None
        self.historia = {est: [] for est in ESTACIONES}
        self.bicicletas = INITIAL_BIKES.copy()
        self.reabastecimientos = []

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
    
    def run_simulation_for_cost(self, inventario_inicial, hora_inicio, hora_fin):
        """
        Función Objetivo: Ejecuta la simulación y devuelve el número total de reabastecimientos.
        """
        data = GraficarDemanda(float(hora_inicio.strftime("%H.%M")), float(hora_fin.strftime("%H.%M")))
        
        # Pre-calcular demanda y horas
        demanda_por_estacion = {est: data.get_data(est)[1] for est in ESTACIONES}
        x = data.get_data(list(ESTACIONES.keys())[0])[0] # Horas
        
        bicicletas = inventario_inicial.copy()
        reabastecimientos_count = 0

        for hora_idx in range(len(x)):
            hora_actual = x[hora_idx]
            usuarios_salientes = {}
            usuarios_entrantes = {est: 0 for est in ESTACIONES}

            for est in ESTACIONES:
                salida_teorica = int(bicicletas[est] * demanda_por_estacion[est][hora_idx])
                salida = min(salida_teorica, bicicletas[est])
                usuarios_salientes[est] = salida
                bicicletas[est] -= salida
            
            for i, est in enumerate(ESTACIONES):
                salida = usuarios_salientes[est]
                weights = []
                if hora_actual <= 9:
                    weights = matriz_probabilidad_mañana[i]
                elif hora_actual <= 13:
                    weights = matriz_probabilidad_media_tarde[i]
                else:
                    weights = matriz_probabilidad_tarde[i]
                
                for _ in range(salida):
                    destino = random.choices(list(ESTACIONES.keys()), weights=weights, k=1)[0]
                    usuarios_entrantes[destino] += 1
            
            for est in ESTACIONES:
                bicicletas[est] += usuarios_entrantes[est]

            for est in ESTACIONES:
                if bicicletas[est] <= MINIMO_POR_ESTACION:
                    donante = max(bicicletas, key=lambda k: bicicletas[k])
                    if donante != est and bicicletas[donante] > MINIMO_POR_ESTACION:
                        cantidad = random.randint(MINIMO_POR_ESTACION, CANTIDAD_TRANSFERIR)
                        
                        bicicletas[donante] -= cantidad
                        bicicletas[est] += cantidad
                        reabastecimientos_count += 1

        
        return reabastecimientos_count

    def local_search_optimization(self, hora_inicio, hora_fin, max_iter=1000, escenarios = 5):
        """
        Algoritmo de Búsqueda Local simple para encontrar el inventario inicial óptimo.
        """
        estaciones_list = list(ESTACIONES.keys())

        current_inventory = INITIAL_BIKES
        current_cost = self.run_simulation_for_cost(current_inventory, hora_inicio, hora_fin)
        best_inventory = current_inventory
        best_cost = current_cost
        
        st.info(f"Costo inicial (reabastecimientos) con la configuración por defecto: **{best_cost}**")
        progress_bar = st.progress(0)
        
        st.subheader("Buscando configuración óptima...")
        
        for i in range(max_iter):
            donor = random.choice(estaciones_list)
            receiver = random.choice(estaciones_list)
            
            if donor != receiver and current_inventory[donor] > 0 and current_inventory[receiver] < 50:
                
                neighbor_inventory = copy.copy(current_inventory)
                neighbor_inventory[donor] -= 1
                neighbor_inventory[receiver] += 1
                
                # 3. Evaluar el vecino
                sumador = 0
                for j in range(escenarios):
                    sumador += self.run_simulation_for_cost(neighbor_inventory, hora_inicio, hora_fin)
                neighbor_cost = sumador // escenarios
                
                if neighbor_cost <= current_cost:
                    current_inventory = neighbor_inventory
                    current_cost = neighbor_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_inventory = current_inventory
                        st.success(f"🎉 ¡Mejora encontrada en la iteración {i+1}! Nuevo Costo Mínimo: **{best_cost}**")

            progress_bar.progress((i + 1) / max_iter)

        st.subheader("Resultado de la Optimización")
        st.write(f"Costo Mínimo (Reabastecimientos): **{best_cost}**")
        st.write("Inventario Inicial Óptimo Sugerido:")
        st.dataframe(pd.DataFrame(list(best_inventory.items()), columns=["Estación", "Bicicletas Óptimas"]))
        
        return best_inventory

    
    def __generar_boton_optimizacion(self):
        st.subheader("Optimización del rebalanceo")
        if st.button("Optimización del inventario inicial"):
            optimal = self.local_search_optimization(self.hora_inicio, self.hora_fin)

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
        for estacion in ESTACIONES:
            horas, y = data.get_data(estacion)
            self.horas = horas
            demanda_por_estacion[estacion] = [y_i for y_i in y]

        for hora_idx in range(len(self.horas)):
            hora_actual = self.horas[hora_idx]
            usuarios_salientes = {}
            usuarios_entrantes = {estacion: 0 for estacion in ESTACIONES}

            # SALIDAS
            for est in ESTACIONES:
                salida_teorica = int(self.bicicletas[est] * demanda_por_estacion[est][hora_idx])
                salida = min(salida_teorica, self.bicicletas[est])
                usuarios_salientes[est] = salida
                self.bicicletas[est] -= salida
            
            # DISTRIBUCIÓN A DESTINOS
            for i, est in enumerate(ESTACIONES):
                salida = usuarios_salientes[est]
                for _ in range(salida):
                    if hora_actual <= 9:
                        destino = random.choices(list(ESTACIONES.keys()), weights=matriz_probabilidad_mañana[i], k=1)[0]
                    elif hora_actual <= 13:
                        destino = random.choices(list(ESTACIONES.keys()), weights=matriz_probabilidad_media_tarde[i], k=1)[0]
                    else:
                        destino = random.choices(list(ESTACIONES.keys()), weights=matriz_probabilidad_tarde[i], k=1)[0]
                    usuarios_entrantes[destino] += 1
            
            # INVENTARIO ACTUALIZADO
            for est in ESTACIONES:
                self.bicicletas[est] += usuarios_entrantes[est]

            # REABASTECIMIENTOS
            for est in ESTACIONES:
                if self.bicicletas[est] <= MINIMO_POR_ESTACION:
                    # Encontrar estación con más bicicletas
                    donante = max(self.bicicletas, key=lambda k: self.bicicletas[k])
                    if donante != est and self.bicicletas[donante] > MINIMO_POR_ESTACION:
                        cantidad = random.randint(MINIMO_POR_ESTACION, CANTIDAD_TRANSFERIR)
                        self.bicicletas[donante] -= cantidad
                        self.bicicletas[est] += cantidad
                        self.reabastecimientos.append({
                            "Hora": hora_actual,
                            "Estación_reabastecida": est,
                            "Estación_donante": donante,
                            "Cantidad": cantidad
                        })
            
            # GUARDAR HISTORIAL
            for est in ESTACIONES:
                self.historia[est].append(self.bicicletas[est])

    def __evolucion_bicicletas(self):
        st.subheader("Comportamiento del número de bicicletas en cada estación a lo largo del día")
        tabs = st.tabs(ESTACIONES)
        for idx, estacion in enumerate(ESTACIONES):
            print(idx, estacion)
            df = pd.DataFrame({
                'Hora': self.horas,
                "Bicicletas": self.historia[estacion]
            })
            fig = px.line(df, x = "Hora", y = "Bicicletas", title = f"Evolución de bicicletas a lo largo del dia en la estación {estacion}")
            with tabs[idx]:
                st.plotly_chart(fig, use_container_width=True)


    def __graficar_todas_estaciones(self):
        st.subheader("Evolución de bicicletas en todas las estaciones")
        
        df = pd.DataFrame({"Hora": self.horas})
        for est in ESTACIONES:
            df[est] = self.historia[est]

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
                    hora_sim = np.array(self.horas)
                    hora_idx = (np.abs(hora_sim - hora)).argmin()
                    cantidad = self.historia[estacion][hora_idx]

                    color = self.__asignar_color(cantidad)
                    folium.Marker(
                        location=ESTACIONES[estacion],
                        popup=f"{estacion}: {cantidad} bicicletas",
                        icon=folium.Icon(color=color)
                    ).add_to(m)

                components.html(m._repr_html_(), height=500)
    
    def __graficar_reabastecimientos(self):
        if self.reabastecimientos:
            st.subheader(f"⚠️ Reabastecimientos realizados (Total: {len(self.reabastecimientos)})")
            df = pd.DataFrame(self.reabastecimientos)

            df_grouped = df.groupby("Estación_reabastecida")["Cantidad"].count().reset_index()
            df_grouped.rename(columns={"Cantidad": "Reabastecimientos"}, inplace=True)

            tab1, tab2, tab3 = st.tabs(["Tabla", "Barras", "Torta"])

            with tab1:
                st.dataframe(df)

            with tab2:
                fig_bar = px.bar(
                    df_grouped,
                    x="Estación_reabastecida",
                    y="Reabastecimientos",
                    text="Reabastecimientos",
                    color="Estación_reabastecida",
                    title="Reabastecimientos por estación"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with tab3:
                fig_pie = px.pie(
                    df_grouped,
                    names="Estación_reabastecida",
                    values="Reabastecimientos",
                    title="Distribución de reabastecimientos"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("✅ No se requirió ningún reabastecimiento durante la simulación.")


    def run(self):
        self.__generar_sliders()
        self.__generar_boton_optimizacion()
        total = self.__sumar_bicicletas()

        if total > TOTAL_BICICLETAS_OPERATIVAS:
            st.error("Has excedido el límite máximo de bicicletas operativas.")
            return

        st.subheader("Mapa actualizado")
        components.html(self.__dibujar_mapa(), height=500)

        self.__graficar()
        self.__graficar_demanda()
        self.__simular()
        self.__evolucion_bicicletas()
        self.__graficar_todas_estaciones()
        self.__graficar_reabastecimientos()
        self.__generate_maps()
        

app = SimulacionBicirrun()
app.run()
