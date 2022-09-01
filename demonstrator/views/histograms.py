from dash import dcc, Input, Output, State, html
import dash_cytoscape as cyto
import plotly.express as px

from data_manager import FilterOptions, get_histogram_data



def layout():
    return dcc.Loading([
        html.Div(id = 'histogram_container', children=[], className="histogramContainer")
    ])

def title():
    return 'Histograms'

def register_callback(app):
    @app.callback(
        Output(component_id='histogram_container', component_property='children'),
        Input(component_id='filter_store', component_property='data'),
    )
    def update_histograms(filter_options):
        filter_options = FilterOptions(**filter_options)
        data = get_histogram_data(filter_options)
        labels = ['Image Quality (UCIQE)', 'Marine growth percentage', 'Depth', 'Similarities', 'Visual similarities']
        x_axes = ['Image Quality (UCIQE)', 'Marine growth percentage', 'Depth', 'Similarities', 'Visual similarities']
        y_axes = ['#Frames', '#Mosaics', '#Frames', '#Frames', '#Frames']
        histograms = []
        for data, label, y_axis, x_axis in zip(data, labels, y_axes, x_axes):
            fig = px.histogram(data, labels=[label], color_discrete_sequence=["darkgray"])
            fig.update_layout(
                title_text=label, # title of plot
                xaxis_title_text=x_axis, # xaxis label
                yaxis_title_text=y_axis, # yaxis label
                bargap=0.1, # gap between bars of adjacent location coordinates
                showlegend=False,
                hovermode=False
            )
            histograms.append(dcc.Graph(figure=fig))
        return histograms
