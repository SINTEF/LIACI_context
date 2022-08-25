from dash import dcc, Input, Output, State, html, dash_table
import dash_cytoscape as cyto
import plotly.express as px




def layout():

    columns = [
        {'name': 'Image', 'id': 'image', "presentation": "markdown"},
        {'name': 'Quality', 'id': 'quality'}
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
    return html.Div(style={
        'max-height': '80vh',
        'overflow':'auto'
    },children=[
        dash_table.DataTable(id='image_table', columns=columns, **table_options),
    ])

def title():
    return 'Images'

def register_callback(app):
    @app.callback(
        Output(component_id='image_table', component_property='data'),
        Input(component_id='data_store', component_property='data'),
    )
    def update_image_table(data):
        return data['image_table']
