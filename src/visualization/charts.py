import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict

class ChartVisualizer:
    """Create interactive charts for deforestation analysis"""
    
    def __init__(self, theme='plotly_dark'):
        self.theme = theme
    
    def create_trend_chart(self, years: List[int], values: List[float], 
                          title: str = "Deforestation Trend",
                          y_label: str = "Area (km²)"):
        """
        Create line chart for trend analysis
        
        Args:
            years: List of years
            values: List of values
            title: Chart title
            y_label: Y-axis label
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            mode='lines+markers',
            name='Deforestation',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10, color='#c0392b')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title=y_label,
            template=self.theme,
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_pie_chart(self, labels: List[str], values: List[float],
                        title: str = "Distribution"):
        """
        Create pie chart
        
        Args:
            labels: Category labels
            values: Values for each category
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = px.pie(
            values=values,
            names=labels,
            title=title,
            color_discrete_sequence=px.colors.sequential.Greens_r
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>%{value} km²<br>%{percent}'
        )
        
        fig.update_layout(
            template=self.theme,
            height=400
        )
        
        return fig
    
    def create_bar_chart(self, categories: List[str], values: List[float],
                        title: str = "Comparison",
                        x_label: str = "Category",
                        y_label: str = "Value"):
        """
        Create bar chart
        
        Args:
            categories: Category names
            values: Values for each category
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=values,
                marker_color=px.colors.sequential.Greens_r,
                text=values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.theme,
            height=400
        )
        
        return fig
    
    def create_heatmap(self, data: pd.DataFrame, title: str = "Heatmap"):
        """
        Create heatmap
        
        Args:
            data: DataFrame with data
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdYlGn_r',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            template=self.theme,
            height=500
        )
        
        return fig
    
    def create_gauge_chart(self, value: float, title: str = "Risk Level",
                          max_value: float = 100):
        """
        Create gauge chart for risk assessment
        
        Args:
            value: Current value
            title: Chart title
            max_value: Maximum value
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': max_value * 0.5},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': "darkred" if value > 70 else "orange" if value > 40 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, max_value], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        
        fig.update_layout(
            template=self.theme,
            height=300
        )
        
        return fig
    
    def create_multi_line_chart(self, data: Dict[str, List], years: List[int],
                               title: str = "Multi-Series Comparison"):
        """
        Create multi-line chart
        
        Args:
            data: Dictionary with series name as key and values as list
            years: List of years (x-axis)
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i, (name, values) in enumerate(data.items()):
            fig.add_trace(go.Scatter(
                x=years,
                y=values,
                mode='lines+markers',
                name=name,
                line=dict(width=2, color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title='Value',
            template=self.theme,
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_area_chart(self, years: List[int], values: List[float],
                         title: str = "Area Chart"):
        """
        Create area chart
        
        Args:
            years: List of years
            values: List of values
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=values,
            fill='tozeroy',
            mode='lines',
            line=dict(color='#27ae60', width=2),
            fillcolor='rgba(39, 174, 96, 0.3)'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Year',
            yaxis_title='Area (km²)',
            template=self.theme,
            height=400
        )
        
        return fig
