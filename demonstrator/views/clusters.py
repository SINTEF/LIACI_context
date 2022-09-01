import json
import uuid
from dash import dcc, Input, Output, State, html, dash_table
import dash_cytoscape as cyto
import plotly.express as px
from data_manager import FilterOptions, get_cluster_table, get_frames_cluster



def layout():

    columns = [
        {'name': 'ID', 'id': 'cluster'},
        {'name': 'Key Frame', 'id': 'key_frame', "presentation": "markdown"},
        {'name': 'Key Words', 'id': 'keywords'},
    ]


    table_options = {
        'style_cell_conditional':[
            {
                'if': {'column_id': 'name'},
                'textAlign': 'left'
            }
        ],
        'style_cell':{
            'padding': '5px',
            'font-size': '12px'
            },
        'style_header':{
            'fontWeight': 'bold',
            'font-size': '12px',
            'font-family': '"Open Sans", sans-serif'
        },
        'sort_action':"native",
        'sort_by': [{'column_id': 'quality', 'direction': 'desc'}]
    }
    return html.Div(className="two_column", children=[dcc.Loading(
        [html.Div(className="graphContainer", id="clusters_table_container", children=[dash_table.DataTable(id='clusters_table', columns=columns, **table_options)]),
        html.Div(className="nodeInfo", style={'width':'25%'}, id="cluster_info", children=[])
        ])
        ])

def title():
    return 'Clusters'

def register_callback(app):
    @app.callback(
        Output(component_id='clusters_table', component_property='data'),
        Input(component_id='filter_store', component_property='data'),
    )
    def update_image_table(filter_options):
        filter_options = FilterOptions(**filter_options)
        cluster_table = get_cluster_table(filter_options)
        return cluster_table

    @app.callback(
        Output(component_id='cluster_info', component_property='children'),
        State(component_id='cluster_info', component_property='children'),
        Input(component_id='clusters_table', component_property='active_cell'),
    )
    def update_cluster_info(old,  data):
        if data is None:
            return ["Select cluster to show frames"]
        row = data["row_id"]
        if not hasattr(update_cluster_info, 'last_row'):
            update_cluster_info.last_row = -1
        if update_cluster_info.last_row != row:
            update_cluster_info.last_row = row
            inspection_id = row.split('=')[1]
            cluster_id = row.split('=')[0]
            frames = get_frames_cluster(inspection_id, cluster_id) 
            return [html.Div([html.P(), html.Img(src=f"/assets/imgs/{'mosaics' if f['path'].startswith('m') else 'frames'}/{f['path']}", style={'width':'100%'})]) for f in frames]
        else:
            return old
