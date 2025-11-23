import folium
from folium import plugins
from typing import List, Tuple, Dict
import pandas as pd

class MapVisualizer:
    """Create interactive geospatial maps for deforestation monitoring"""
    
    def __init__(self, center=[-3.4653, -62.2159], zoom=6):
        self.center = center
        self.zoom = zoom
    
    def create_base_map(self, tiles='OpenStreetMap'):
        """
        Create base map
        
        Args:
            tiles: Map tile style
            
        Returns:
            Folium map object
        """
        m = folium.Map(
            location=self.center,
            zoom_start=self.zoom,
            tiles=tiles
        )
        
        return m
    
    def add_marker(self, map_obj, location: Tuple[float, float], 
                   popup_text: str, icon_color: str = 'blue',
                   icon: str = 'info-sign'):
        """
        Add marker to map
        
        Args:
            map_obj: Folium map object
            location: (latitude, longitude)
            popup_text: Text for popup
            icon_color: Color of the icon
            icon: Icon type
        """
        folium.Marker(
            location=location,
            popup=popup_text,
            icon=folium.Icon(color=icon_color, icon=icon)
        ).add_to(map_obj)
    
    def add_circle_marker(self, map_obj, location: Tuple[float, float],
                         radius: float, popup_text: str, color: str = 'red'):
        """
        Add circle marker to map
        
        Args:
            map_obj: Folium map object
            location: (latitude, longitude)
            radius: Circle radius in meters
            popup_text: Text for popup
            color: Circle color
        """
        folium.Circle(
            location=location,
            radius=radius,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.4
        ).add_to(map_obj)
    
    def add_heatmap(self, map_obj, locations: List[Tuple[float, float]],
                   intensities: List[float] = None):
        """
        Add heatmap layer
        
        Args:
            map_obj: Folium map object
            locations: List of (latitude, longitude) tuples
            intensities: Optional intensity values for each location
        """
        if intensities:
            heat_data = [[loc[0], loc[1], intensity] 
                        for loc, intensity in zip(locations, intensities)]
        else:
            heat_data = locations
        
        plugins.HeatMap(heat_data).add_to(map_obj)
    
    def add_polygon(self, map_obj, coordinates: List[Tuple[float, float]],
                   popup_text: str = None, color: str = 'red'):
        """
        Add polygon to map
        
        Args:
            map_obj: Folium map object
            coordinates: List of (latitude, longitude) tuples
            popup_text: Text for popup
            color: Polygon color
        """
        folium.Polygon(
            locations=coordinates,
            popup=popup_text,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.3
        ).add_to(map_obj)
    
    def create_deforestation_map(self, alert_zones: List[Dict]):
        """
        Create map with deforestation alert zones
        
        Args:
            alert_zones: List of dictionaries with zone information
                        Each dict should have: lat, lon, name, severity, area
            
        Returns:
            Folium map object
        """
        m = self.create_base_map()
        
        # Color mapping for severity
        severity_colors = {
            'Critical': 'darkred',
            'High': 'red',
            'Medium': 'orange',
            'Low': 'yellow',
            'Minimal': 'green'
        }
        
        for zone in alert_zones:
            lat = zone.get('lat')
            lon = zone.get('lon')
            name = zone.get('name', 'Alert Zone')
            severity = zone.get('severity', 'Medium')
            area = zone.get('area', 0)
            
            color = severity_colors.get(severity, 'blue')
            
            # Create popup HTML
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="margin: 0; color: {color};">{name}</h4>
                <hr style="margin: 5px 0;">
                <p style="margin: 5px 0;"><b>Severity:</b> {severity}</p>
                <p style="margin: 5px 0;"><b>Area:</b> {area} km²</p>
            </div>
            """
            
            # Add marker
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                icon=folium.Icon(color=color, icon='warning-sign')
            ).add_to(m)
            
            # Add circle to show affected area
            if area > 0:
                radius = area * 100  # Scale for visualization
                self.add_circle_marker(m, [lat, lon], radius, popup_html, color)
        
        return m
    
    def add_layer_control(self, map_obj):
        """
        Add layer control to map
        
        Args:
            map_obj: Folium map object
        """
        folium.LayerControl().add_to(map_obj)
    
    def add_fullscreen_button(self, map_obj):
        """
        Add fullscreen button to map
        
        Args:
            map_obj: Folium map object
        """
        plugins.Fullscreen().add_to(map_obj)
    
    def add_minimap(self, map_obj):
        """
        Add minimap to map
        
        Args:
            map_obj: Folium map object
        """
        minimap = plugins.MiniMap()
        map_obj.add_child(minimap)
    
    def create_choropleth_map(self, geo_data, data: pd.DataFrame,
                             key_column: str, value_column: str):
        """
        Create choropleth map
        
        Args:
            geo_data: GeoJSON data
            data: DataFrame with values
            key_column: Column name for joining
            value_column: Column name for coloring
            
        Returns:
            Folium map object
        """
        m = self.create_base_map()
        
        folium.Choropleth(
            geo_data=geo_data,
            data=data,
            columns=[key_column, value_column],
            key_on=f'feature.properties.{key_column}',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=value_column
        ).add_to(m)
        
        return m
