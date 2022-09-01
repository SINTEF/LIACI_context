from dash import dcc, Input, Output, State, html, dash_table
import dash_cytoscape as cyto
import plotly.express as px

from data_manager import FilterOptions, get_tables




def layout():

    return dcc.Loading([html.Div(id = "table_tab_content", children=[])])

def title():
    return 'Tables'

def register_callback(app):
    @app.callback(
        Output(component_id='table_tab_content', component_property='children'),
        Input(component_id='filter_store', component_property='data'),
    )
    def update_tables(filter_options):
        ship_colums = [{'name': 'Ship name', 'id': 'name'}]
        part_colums = [{'name': 'Part', 'id': 'name'},]
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
        filter_options = FilterOptions(**filter_options)
        by_ship_table_data, by_part_table_data = get_tables(filter_options)
        return [
            html.H1('By ship aggregation'),
            html.Br(),
            dash_table.DataTable(id='by_ship_table', columns=ship_colums + defect_columns, **table_options, data=by_ship_table_data),
            html.Br(),
            html.Br(),
            html.H1('By part aggregation'),
            html.Br(),
            dash_table.DataTable(id='by_part_table', columns=part_colums + defect_columns, **table_options, data=by_part_table_data)
        ]

