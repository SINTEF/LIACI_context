from dash import dcc, Input, Output, State, html
import dash_cytoscape as cyto
import plotly.express as px



def layout():
    return html.Div([
        dcc.Graph(id="q_u_hist"),
        dcc.Graph(id="d_hist"),
        dcc.Graph(id="mg_hist"),
        dcc.Graph(id="sim_hist"),
        dcc.Graph(id="vsim_hist")
    ])

def title():
    return 'Histograms'

def register_callback(app):
    @app.callback(
        Output(component_id='q_u_hist', component_property='figure'),
        Input(component_id='data_store', component_property='data'),
    )
    def update_histograms(data):
        fig = px.histogram(data['q_hist'], range_x=(0,16), labels=["Image Quality (UCIQE)"], color_discrete_sequence=["darkgray"])
        fig.update_layout(
            title_text='Image Quality (UCIQE)', # title of plot
            xaxis_title_text='Image Quality (UCIQE)', # xaxis label
            yaxis_title_text='#Frames', # yaxis label
            bargap=0.1, # gap between bars of adjacent location coordinates
            showlegend=False,
            hovermode=False
        )
        return fig

    @app.callback(
        Output(component_id='mg_hist', component_property='figure'),
        Input(component_id='data_store', component_property='data'),
    )
    def update_histograms(data):
        fig = px.histogram(data['mg_hist'], labels=["Marine Growth Percentage"], color_discrete_sequence=["darkgray"])
        fig.update_layout(
            title_text='Marine Growth Percentage', # title of plot
            xaxis_title_text='Marine Growth Percentage', # xaxis label
            yaxis_title_text='#Mosaics', # yaxis label
            bargap=0.1, # gap between bars of adjacent location coordinates
            showlegend=False,
            hovermode=False
        )
        return fig

    @app.callback(
        Output(component_id='vsim_hist', component_property='figure'),
        Input(component_id='data_store', component_property='data'),
    )
    def update_histograms(data):
        fig = px.histogram(data['vsim_hist'], labels=['Visual Similarities'], color_discrete_sequence=["darkgray"])
        fig.update_layout(
            title_text='Visual Similarities', # title of plot
            xaxis_title_text='Visual Similarities', # xaxis label
            yaxis_title_text=#Frames'', # yaxis label
            bargap=0.1, # gap between bars of adjacent location coordinates
            showlegend=False,
            hovermode=False
        )
        return fig
    @app.callback(
        Output(component_id='sim_hist', component_property='figure'),
        Input(component_id='data_store', component_property='data'),
    )
    def update_histograms(data):
        fig = px.histogram(data['sim_hist'], labels=['Similarities'], color_discrete_sequence=["darkgray"])
        fig.update_layout(
            title_text='Similarities', # title of plot
            xaxis_title_text='Similarities', # xaxis label
            yaxis_title_text='#Similarities', # yaxis label
            bargap=0.1, # gap between bars of adjacent location coordinates
            showlegend=False,
            hovermode=False
        )
        return fig
    @app.callback(
        Output(component_id='d_hist', component_property='figure'),
        Input(component_id='data_store', component_property='data'),
    )
    def update_histograms(data):
        fig = px.histogram(data['d_hist'], range_x=(-10,0), labels=['Depth'], color_discrete_sequence=["darkgray"])
        fig.update_layout(
            title_text='Depth', # title of plot
            xaxis_title_text='Depth', # xaxis label
            yaxis_title_text='#Frames', # yaxis label
            bargap=0.1, # gap between bars of adjacent location coordinates
            showlegend=False,
            hovermode=False
        )
        return fig