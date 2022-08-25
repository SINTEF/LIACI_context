from dash import dcc, Input, Output, State, html, dash_table
import dash_cytoscape as cyto
import plotly.express as px




def layout():

    ship_colums = [
        {'name': 'Ship name', 'id': 'name'},
    ]
    part_colums = [
        {'name': 'Part', 'id': 'name'},
    ]
    defect_columns=[
        {'name': 'paint peel', 'id': 'paint_peel'},
        {'name': 'marine growth', 'id': 'marine_growth'},
        {'name': 'corrosion', 'id': 'corrosion'},
        {'name': 'defect', 'id': 'defect'},
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
    }
    return html.Div([
        html.H1('By ship aggregation'),
        html.Br(),
        dash_table.DataTable(id='by_ship_table', columns=ship_colums + defect_columns, **table_options),
        html.Br(),
        html.Br(),
        html.H1('By part aggregation'),
        html.Br(),
        dash_table.DataTable(id='by_part_table', columns=part_colums + defect_columns, **table_options)
    ])

def title():
    return 'Tables'

def register_callback(app):
    @app.callback(
        Output(component_id='by_ship_table', component_property='data'),
        Input(component_id='data_store', component_property='data'),
    )
    def update_ship_table(data):

        return data['by_ship_table']

    @app.callback(
        Output(component_id='by_part_table', component_property='data'),
        Input(component_id='data_store', component_property='data'),
    )
    def update_part_table(data):

        return data['by_part_table']
