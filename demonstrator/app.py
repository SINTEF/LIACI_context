from atexit import register
from dataclasses import replace
from platform import node
import dash
import dash_cytoscape as cyto
from dash import dcc, Input, Output, State, html
import py2neo
from py2neo import NodeMatcher, Graph, RelationshipMatcher

import networkx as nx
import views.graph as graph_view
import views.histograms as hist_view
import views.tables as tables_view
import views.controls as controls
import views.images as images_view
import views.bar_charts as bar_charts_view
import views.clusters as clusters_view
import data_manager

import logging
logging.basicConfig(level=logging.WARN)

sites = [graph_view, clusters_view, hist_view, tables_view, images_view, bar_charts_view]

external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    "/assets/style.css",
    "https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css"
]

app = dash.Dash('LIACi Context Demonstrator', external_stylesheets=external_stylesheets)



############################################################################################################################
# LAYOUT                                                                                                                   #
############################################################################################################################
app.layout = html.Div(className='main_page_container', children=[
    dcc.Store(id='filter_store'),
    dcc.Store(id='data_store'),
    html.Div(className='header', children=[
        html.Img(src='assets/SintefLogo.png'),
        html.Img(src='assets/LiaciContextLogo.png')
    ]),
    html.Div(className='two_columns', children=[
        html.Div(className='left', children = [controls.layout()]),
        html.Div(className='right', children = [
           dcc.Tabs(id='tabs', value='0', children=[dcc.Tab(label=s.title(), value=str(i)) for i,s in enumerate(sites)]) ,
           html.Div(id='tabs-content')
        ])])
    ])


############################################################################################################################
# CALLBACKS                                                                                                                #
############################################################################################################################
@app.callback(
    Output('tabs-content', 'children'), 
    Input('tabs', 'value'))
def render_tabs_content(tab):
    return sites[int(tab)].layout()


data_manager.register_callbacks(app)
controls.register_callbacks(app)

for site in sites:
    site.register_callback(app)






if __name__ == '__main__':
    app.run_server(debug=True, port='8051')

