import dash
from dash import dcc, Input, Output, html

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

sites = [graph_view, clusters_view, hist_view, tables_view, bar_charts_view]

external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    "/assets/style.css",
    "https://cdn.jsdelivr.net/npm/bootstrap@5.0.0-beta1/dist/css/bootstrap.min.css"
]

app = dash.Dash('LIACi Context Demonstrator', external_stylesheets=external_stylesheets)
app.title='LIACi Context Demonstrator'


############################################################################################################################
# LAYOUT                                                                                                                   #
############################################################################################################################
app.layout = html.Div(className='main_page_container', children=[
    dcc.Store(id='filter_store'),
    dcc.Store(id='data_store'),
    html.Div(className='header', children=[
        html.Img(src='assets/sintef_logo.png'),
        html.Img(src='assets/LiaciContextLogo.png')
    ]),
    html.Div(className='two_columns', children=[
        html.Div(className='left', children = [controls.layout(),
        
    html.Div(className='footer', children=[
        html.P([html.B("Publication Date"),": 30.08.2022"]),
        html.P([html.B("Grants"),": The dataset contained on this page was created within the ", html.A(["LIACi"], href="https://www.sintef.no/en/projects/2021/liaci/"), " project. LIACi was funded by the Research Council of Norway under the project No 317854."]),
        html.P([html.B("License"),": This work and the data is restricted by the ", html.A(["Creative Commons Attribition Non Commercial Share Alike 4.0 International"], href="https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"), " license."]),
    ]),
        
        ]),
        html.Div(className='right', children = [
           dcc.Tabs(id='tabs', value='0', children=[dcc.Tab(label=s.title(), value=str(i)) for i,s in enumerate(sites)]) ,
           html.Div(id='tabs-content')
        ])]),
    ])


############################################################################################################################
# CALLBACKS                                                                                                                #
############################################################################################################################
@app.callback(
    Output('tabs-content', 'children'), 
    Input('tabs', 'value'))
def render_tabs_content(tab):
    return sites[int(tab)].layout()


controls.register_callbacks(app)

for site in sites:
    site.register_callback(app)






app.run_server('0.0.0.0', debug=False, port='8051')

