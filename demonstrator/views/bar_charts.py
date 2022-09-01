import json
from dash import dcc, Input, Output, State, html, dash_table, ALL
import dash_cytoscape as cyto
import plotly.express as px
from plotly import graph_objects as go

from data_manager import FilterOptions, get_frames_angle, get_headings_hist




def layout():

    
    return dcc.Loading([
        html.Div(className="two_columns", children=[
            html.Div(className="graphContainer",id="img_direction"),
            html.Div(className="nodeInfo", id="frameList")
        ])
    ])

def title():
    return 'Bar charts'

def register_callback(app):

    g_to_inspection = []
    @app.callback(
        Output(component_id='frameList', component_property='children'),
        State(component_id='filter_store', component_property= 'data'),
        Input(component_id={'type':'heading_graph', 'index': ALL}, component_property='clickData'),
    )
    def update_nodes_table(filter_store, click_data):
        filter_options = FilterOptions(**filter_store)
        if not hasattr(update_nodes_table, 'old_data'):
            update_nodes_table.old_data = click_data
        
        changed = None
        changed_angle = None
        for i, d in enumerate(click_data):
            if d is not None and (i >= len(update_nodes_table.old_data) or json.dumps(d) != json.dumps(update_nodes_table.old_data[i])):
                changed = g_to_inspection[i]
                changed_angle = d['points'][0]['theta']

        if changed is not None:
           frames = get_frames_angle(changed, (changed_angle + 360) % 360, filter_options) 
           return [html.Img(src=f"/assets/imgs/{'mosaics' if f['path'].startswith('m') else 'frames'}/{f['path']}", style={'width':'100%'}) for f in frames]

        update_nodes_table.old_data = click_data
        return "Select sector to show frames"
    
    @app.callback(
        Output(component_id='img_direction', component_property='children'),
        Input(component_id='filter_store', component_property='data'),
    )
    def update_image_table(filter_options):
        filter_options = FilterOptions(**filter_options)
        heading_hist = get_headings_hist(filter_options)
        graphs = []
        for inspection_id, heading_hist in heading_hist.items():
            g_to_inspection.append(inspection_id)
            title=f"{heading_hist['ship_name']} on {heading_hist['date']}"
            fig = px.bar_polar(heading_hist['data'], r="count", theta="heading", width=800, height=800, title=title)
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor="middle",  
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(
                    size=14)),
                polar=dict(hole=0.4, radialaxis=dict(showticklabels=True, ticks='', linewidth=0)
                        ),
                margin=dict(t=110),
            )
            fig.add_shape(type="path",
                xsizemode="pixel",
                ysizemode="pixel",
                xanchor="0.5",
                yanchor="0.5",  
            
                path="M 0 -100 L 40 -100 C 60 0 60 60 0 100 C -60 60 -60 0 -40 -100 Z",
                fillcolor="LightGray",
                line_color="Gray"
            )
            graphs.append(dcc.Graph(id={
                'type': 'heading_graph',
                'index': inspection_id
            }, figure=fig))


        return graphs
