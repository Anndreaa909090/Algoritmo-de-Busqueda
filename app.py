import streamlit as st
import math
import heapq
from typing import Dict, List, Tuple, Optional
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go

CUENCA_NODES: Dict[str, Dict[str, float]] = {
    "Catedral Nueva": {"lat": -2.8975, "lon": -79.005, "descripcion": "Centro histórico de Cuenca"},
    "Parque Calderón": {"lat": -2.89741, "lon": -79.00438, "descripcion": "Corazón de Cuenca"},
    "Puente Roto": {"lat": -2.90423, "lon": -79.00142, "descripcion": "Monumento histórico"},
    "Museo Pumapungo": {"lat": -2.90607, "lon": -78.99681, "descripcion": "Museo de antropología"},
    "Terminal Terrestre": {"lat": -2.89222, "lon": -78.99277, "descripcion": "Terminal de autobuses"},
    "Mirador de Turi": {"lat": -2.92583, "lon": -79.0040, "descripcion": "Mirador con vista panorámica"},
    "Plaza San Sebastián": {"lat": -2.8991, "lon": -79.0085, "descripcion": "Plaza histórica"},
    "Mercado 10 de Agosto": {"lat": -2.8942, "lon": -79.0038, "descripcion": "Mercado tradicional"},
    "Museo de las Conceptas": {"lat": -2.8965, "lon": -79.0072, "descripcion": "Museo de arte religioso"},
    "Parque de la Madre": {"lat": -2.8918, "lon": -79.0081, "descripcion": "Parque urbano"},
    "Catedral Vieja": {"lat": -2.8968, "lon": -79.0049, "descripcion": "Iglesia del Sagrario"},
    "Barranco del Tomebamba": {"lat": -2.9005, "lon": -79.0028, "descripcion": "Área natural"},
    "Museo Municipal de Arte Moderno": {"lat": -2.8983, "lon": -79.0067, "descripcion": "Museo de arte moderno"},
    "Parque El Paraíso": {"lat": -2.8902, "lon": -79.0015, "descripcion": "Parque recreativo"},
    "Iglesia de Santo Domingo": {"lat": -2.8956, "lon": -79.0063, "descripcion": "Iglesia histórica"},
    "Mirador de Cullca": {"lat": -2.9187, "lon": -79.0123, "descripcion": "Mirador natural"}
}

GRAPH_EDGES = {
    "Catedral Nueva": ["Parque Calderón", "Puente Roto", "Museo Pumapungo", "Plaza San Sebastián", "Catedral Vieja"],
    "Parque Calderón": ["Catedral Nueva", "Terminal Terrestre", "Puente Roto", "Mercado 10 de Agosto", "Catedral Vieja"],
    "Puente Roto": ["Catedral Nueva", "Parque Calderón", "Museo Pumapungo", "Mirador de Turi", "Barranco del Tomebamba"],
    "Museo Pumapungo": ["Catedral Nueva", "Puente Roto", "Terminal Terrestre", "Barranco del Tomebamba"],
    "Terminal Terrestre": ["Parque Calderón", "Museo Pumapungo", "Mirador de Turi", "Parque El Paraíso"],
    "Mirador de Turi": ["Puente Roto", "Terminal Terrestre", "Mirador de Cullca"],
    "Plaza San Sebastián": ["Catedral Nueva", "Museo de las Conceptas", "Parque de la Madre"],
    "Mercado 10 de Agosto": ["Parque Calderón", "Parque de la Madre", "Iglesia de Santo Domingo"],
    "Museo de las Conceptas": ["Plaza San Sebastián", "Iglesia de Santo Domingo", "Museo Municipal de Arte Moderno"],
    "Parque de la Madre": ["Plaza San Sebastián", "Mercado 10 de Agosto", "Parque El Paraíso"],
    "Catedral Vieja": ["Catedral Nueva", "Parque Calderón", "Museo Municipal de Arte Moderno"],
    "Barranco del Tomebamba": ["Puente Roto", "Museo Pumapungo", "Museo Municipal de Arte Moderno"],
    "Museo Municipal de Arte Moderno": ["Museo de las Conceptas", "Catedral Vieja", "Barranco del Tomebamba"],
    "Parque El Paraíso": ["Terminal Terrestre", "Parque de la Madre", "Mirador de Cullca"],
    "Iglesia de Santo Domingo": ["Mercado 10 de Agosto", "Museo de las Conceptas"],
    "Mirador de Cullca": ["Mirador de Turi", "Parque El Paraíso"]
}



def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia Haversine en km."""
    R = 6371.0
    
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def euclidean_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia euclidiana aproximada en km (111 km ≈ 1°)."""
    return math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2) * 111.0


class AStarPathFinder:
    def __init__(self, nodes: Dict, edges: Dict):
        self.nodes = nodes
        self.edges = edges
        self.explored: List[str] = []

    def heuristic(self, node: str, goal: str) -> float:
        n, g = self.nodes[node], self.nodes[goal]
        return euclidean_distance(n["lat"], n["lon"], g["lat"], g["lon"])

    def get_distance(self, node1: str, node2: str) -> float:
        n1, n2 = self.nodes[node1], self.nodes[node2]
        return haversine_distance(n1["lat"], n1["lon"], n2["lat"], n2["lon"])

    def find_path(self, start: str, goal: str) -> Tuple[Optional[List[str]], float, int]:
        self.explored = []
        frontier = []
        counter = 0
        
        heapq.heappush(frontier, (0.0, counter, start, [start], 0.0))
        visited = set()
        
        while frontier:
            f_score, _, current, path, g_score = heapq.heappop(frontier)
            
            if current in visited:
                continue
                
            visited.add(current)
            self.explored.append(current)
            
            if current == goal:
                return path, g_score, len(self.explored)
            
            for neighbor in self.edges.get(current, []):
                if neighbor in visited:
                    continue
                    
                edge_cost = self.get_distance(current, neighbor)
                new_g = g_score + edge_cost
                h = self.heuristic(neighbor, goal)
                
                counter += 1
                heapq.heappush(frontier, (new_g + h, counter, neighbor, path + [neighbor], new_g))
        
        return None, float("inf"), len(self.explored)


def bfs_search(nodes: Dict, edges: Dict, start: str, goal: str) -> Tuple[Optional[List[str]], float, int]:
    from collections import deque
    
    visited = set()
    queue = deque([(start, [start], 0.0)])
    explored = 0
    
    while queue:
        current, path, distance = queue.popleft()
        
        if current in visited:
            continue
            
        visited.add(current)
        explored += 1
        
        if current == goal:
            return path, distance, explored
            
        for neighbor in edges.get(current, []):
            if neighbor not in visited:
                edge_distance = haversine_distance(
                    nodes[current]["lat"], nodes[current]["lon"],
                    nodes[neighbor]["lat"], nodes[neighbor]["lon"]
                )
                queue.append((neighbor, path + [neighbor], distance + edge_distance))
    
    return None, float("inf"), explored

def dfs_search(nodes: Dict, edges: Dict, start: str, goal: str) -> Tuple[Optional[List[str]], float, int]:
    stack = [(start, [start], 0.0)]
    visited = set()
    explored = 0
    
    while stack:
        current, path, distance = stack.pop()
        
        if current in visited:
            continue
            
        visited.add(current)
        explored += 1
        
        if current == goal:
            return path, distance, explored
            
        for neighbor in reversed(edges.get(current, [])):
            if neighbor not in visited:
                edge_distance = haversine_distance(
                    nodes[current]["lat"], nodes[current]["lon"],
                    nodes[neighbor]["lat"], nodes[neighbor]["lon"]
                )
                stack.append((neighbor, path + [neighbor], distance + edge_distance))
    
    return None, float("inf"), explored


def crear_mapa_interactivo(nodes, camino, inicio, destino, algoritmo):
    """Crea un mapa interactivo con Plotly"""
    
    
    lats = [node["lat"] for node in nodes.values()]
    lons = [node["lon"] for node in nodes.values()]
    nombres = list(nodes.keys())
    descripciones = [node["descripcion"] for node in nodes.values()]
    
   
    fig = go.Figure()
    
    
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=dict(size=10, color='lightgray'),
        text=[f"{n}<br>{d}" for n, d in zip(nombres, descripciones)],
        hoverinfo='text',
        name='Puntos de Interés',
        showlegend=False
    ))
    
   
    fig.add_trace(go.Scattermapbox(
        lat=[nodes[inicio]["lat"]],
        lon=[nodes[inicio]["lon"]],
        mode='markers+text',
        marker=dict(size=20, color='green'),
        text=["INICIO"],
        textposition="top center",
        name='Inicio',
        hoverinfo='text',
        hovertext=f"INICIO: {inicio}<br>{nodes[inicio]['descripcion']}"
    ))
    
    
    fig.add_trace(go.Scattermapbox(
        lat=[nodes[destino]["lat"]],
        lon=[nodes[destino]["lon"]],
        mode='markers+text',
        marker=dict(size=20, color='red'),
        text=["DESTINO"],
        textposition="top center",
        name='Destino',
        hoverinfo='text',
        hovertext=f"DESTINO: {destino}<br>{nodes[destino]['descripcion']}"
    ))
    
    if camino:
        ruta_lats = [nodes[punto]["lat"] for punto in camino]
        ruta_lons = [nodes[punto]["lon"] for punto in camino]
        ruta_nombres = [f"Paso {i+1}: {punto}" for i, punto in enumerate(camino)]
        
        fig.add_trace(go.Scattermapbox(
            lat=ruta_lats,
            lon=ruta_lons,
            mode='lines+markers',
            line=dict(width=4, color='blue'),
            marker=dict(size=8, color='blue'),
            text=ruta_nombres,
            hoverinfo='text',
            name='Ruta Óptima'
        ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=nodes[inicio]["lat"], lon=nodes[inicio]["lon"]),
            zoom=13
        ),
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"Mapa de Ruta en Cuenca - Algoritmo: {algoritmo}",
        showlegend=True
    )
    
    return fig


def main():
    st.set_page_config(page_title="Rutas Óptimas en Cuenca", page_icon="🗺️", layout="wide")
    
    st.title("🗺️ Algoritmo A* para Optimización de Rutas en Cuenca")
    st.markdown("---")

    
    with st.sidebar:
        st.header("Configuración de Búsqueda")
        
        algoritmo = st.selectbox(
            "Selecciona el algoritmo de búsqueda:",
            ["A*", "BFS (Búsqueda en Amplitud)", "DFS (Búsqueda en Profundidad)"],
            help="Compara la eficiencia de diferentes algoritmos"
        )
        
       
        lugares = list(CUENCA_NODES.keys())
        inicio = st.selectbox("Punto de inicio:", lugares, index=0)
        destino = st.selectbox("Punto de destino:", lugares, index=5)
        
        st.markdown("---")
       
    
   
    pathfinder = AStarPathFinder(CUENCA_NODES, GRAPH_EDGES)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Visualización de la Ruta")
        
        if st.button("Encontrar Ruta Óptima", type="primary"):
            with st.spinner("Buscando la mejor ruta..."):
                start_time = time.time()
                
                if algoritmo == "A*":
                    camino, distancia, nodos_explorados = pathfinder.find_path(inicio, destino)
                elif algoritmo == "BFS (Búsqueda en Amplitud)":
                    camino, distancia, nodos_explorados = bfs_search(CUENCA_NODES, GRAPH_EDGES, inicio, destino)
                else:  
                    camino, distancia, nodos_explorados = dfs_search(CUENCA_NODES, GRAPH_EDGES, inicio, destino)
                
                execution_time = time.time() - start_time
        
            if camino:
                st.success(f"¡Ruta encontrada en {execution_time:.4f} segundos!")
                
               
                fig = crear_mapa_interactivo(CUENCA_NODES, camino, inicio, destino, algoritmo)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("No se pudo encontrar una ruta entre los puntos seleccionados.")
    
    with col2:
        st.subheader("Resultados de la Búsqueda")
        
        if 'camino' in locals() and camino:
            st.metric("Distancia Total", f"{distancia:.2f} km")
            st.metric("Nodos en la Ruta", len(camino))
            st.metric("Nodos Explorados", nodos_explorados)
            st.metric("Tiempo de Ejecución", f"{execution_time:.4f} s")
            
            st.subheader("Ruta Detallada")
            for i, punto in enumerate(camino):
                emoji = "📍" if i == 0 else "🏁" if i == len(camino)-1 else "➡️"
                st.write(f"{emoji} **{punto}**")
                if i < len(camino)-1:
                    dist_segmento = haversine_distance(
                        CUENCA_NODES[punto]["lat"], CUENCA_NODES[punto]["lon"],
                        CUENCA_NODES[camino[i+1]]["lat"], CUENCA_NODES[camino[i+1]]["lon"]
                    )
                    st.caption(f"  ↳ {dist_segmento:.2f} km →")
        
        st.subheader("Comparación de Algoritmos")
        st.markdown("""
        - **A*:** Más eficiente, usa heurística para guiar la búsqueda
        - **BFS:** Encuentra el camino más corto pero explora más nodos
        - **DFS:** Puede ser más rápido pero no garantiza optimalidad
        """)
    
    st.markdown("---")
    st.subheader("Análisis Comparativo de Algoritmos")
    
    if st.button("Ejecutar Comparativa de Algoritmos"):
        with st.spinner("Realizando análisis comparativo..."):
            resultados = []
            algoritmos = ["A*", "BFS", "DFS"]
            
            pares_prueba = [
                ("Catedral Nueva", "Mirador de Turi"),
                ("Parque Calderón", "Museo Pumapungo"),
                ("Terminal Terrestre", "Plaza San Sebastián")
            ]
            
            for inicio_p, destino_p in pares_prueba:
                fila = {"Inicio": inicio_p, "Destino": destino_p}
                
                for alg in algoritmos:
                    start_time = time.time()
                    
                    if alg == "A*":
                        camino, distancia, explorados = pathfinder.find_path(inicio_p, destino_p)
                    elif alg == "BFS":
                        camino, distancia, explorados = bfs_search(CUENCA_NODES, GRAPH_EDGES, inicio_p, destino_p)
                    else:  # DFS
                        camino, distancia, explorados = dfs_search(CUENCA_NODES, GRAPH_EDGES, inicio_p, destino_p)
                    
                    tiempo = time.time() - start_time
                    
                    fila[f"{alg} - Distancia"] = f"{distancia:.2f} km" if camino else "No encontrado"
                    fila[f"{alg} - Nodos Explorados"] = explorados
                    fila[f"{alg} - Tiempo (s)"] = f"{tiempo:.4f}"
                
                resultados.append(fila)
            
            df_comparativo = pd.DataFrame(resultados)
            st.dataframe(df_comparativo, use_container_width=True)
            
           

if __name__ == "__main__":
    main()