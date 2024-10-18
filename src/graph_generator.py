import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import plotly.io as pio

class GraphGenerator:
    def generate_pie_chart(self, data, title, filename):
        if isinstance(data, pd.Series):
            df = data.reset_index()
            category_column = df.columns[0]
            value_column = df.columns[1]
        elif isinstance(data, pd.DataFrame):
            df = data
            category_column = df.columns[0]
            value_column = df.columns[1]
        else:
            print(f"Warning: Unsupported data type provided for {title}. Skipping chart generation.")
            return None

        if df.empty:
            print(f"Warning: Empty data provided for {title}. Skipping chart generation.")
            return None

        fig = px.pie(df, values=value_column, names=category_column, title=title)
        fig.update_layout(height=600, width=800)
        fig.write_image(filename)
        return filename

    def generate_horizontal_bar_chart(self, data, title, filename, x_axis_label, y_axis_label):
        if isinstance(data, pd.Series):
            df = data.reset_index()
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            print(f"Warning: Unsupported data type provided for {title}. Skipping chart generation.")
            return None

        if df.empty:
            print(f"Warning: Empty data provided for {title}. Skipping chart generation.")
            return None

        category_column = df.columns[0]
        value_column = df.columns[1]

        fig = go.Figure(go.Bar(
            y=df[category_column],
            x=df[value_column],
            orientation='h'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_axis_label,
            yaxis_title=y_axis_label,
            height=600,
            width=800
        )
        
        fig.write_image(filename)
        return filename

    @staticmethod
    def cm_to_pixels(cm, dpi=300):  # Increased DPI for better quality
        return int(cm * dpi / 2.54)

    def generate_cross_environment_treemap(self, cross_env_data, title="Cross-Environment Traffic", width_cm=20, height_cm=10):
        width_px = self.cm_to_pixels(width_cm)
        height_px = self.cm_to_pixels(height_cm)
        
        fig = px.treemap(cross_env_data, 
                         path=['total', 'Source', 'Destination'],
                         values='Connections',
                         title=title,
                         color='Connections',
                         color_continuous_scale='RdYlBu_r',
                         hover_data=['Connections'])
        
        fig.update_traces(textinfo="label+value")
        fig.update_layout(
            width=width_px,
            height=height_px,
            margin=dict(t=50, l=25, r=25, b=25),
            font=dict(size=10)  # Adjust font size as needed
        )
        
        # Increase the image resolution
        pio.kaleido.scope.default_scale = 2.0  # This doubles the resolution
        
        return fig

    def save_plotly_figure(self, fig, output_path):
        fig.write_image(output_path, scale=2.0, format="png")
        return output_path

    def generate_bar_chart(self, data, title, x_label, y_label, width_cm=25, height_cm=15):
        width_px = self.cm_to_pixels(width_cm)
        height_px = self.cm_to_pixels(height_cm)

        # Sort the dictionary by value in descending order
        sorted_data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))

        fig = go.Figure(data=[go.Bar(x=list(sorted_data.keys()), y=list(sorted_data.values()))])
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            width=width_px,
            height=height_px,
            font=dict(size=10),
            margin=dict(l=50, r=50, t=50, b=100)  # Increase bottom margin for labels
        )
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45, tickfont=dict(size=8))
        return fig

    # Add more graph generation methods as needed
