import uuid
from dash import html, dcc, Input, Output
from data_manager import get_inspections, FilterOptions
import dataclasses

def title():
    return 'Controls'

def layout():
    return html.Div(className='controls', children=[
        html.H4('Choose Inspection(s)'),
        dcc.Dropdown(list(get_inspections().keys()), id='inspections_dd', multi=True),
        html.H4('Include'),
        html.P('Telemetry Similarities'),
        dcc.Slider(id="telemetry_similarities", min=0, max=5,  value=1.5),
        html.P(id="sslider_status"),
        html.P('Visual Similarities'),
        dcc.Slider(id="visual_similarities", min=0, max=5, value=2),
        html.P(id="vsslider_status"),
        dcc.Checklist(['mosaics'], id='extras_cl'),
        html.H4('Include the following findings'),
        dcc.Checklist(['marine growth', 'paint peel', 'corrosion', 'defect'], id='defects_cl'),
        html.H4('On the following part'),
        dcc.Checklist(['anode', 'propeller', 'bilge keel', 'sea chest grating', 'over board valves'], id='findings_cl'),
        html.Br(),
        html.Div(id='status')
    ])

def register_callbacks(app):
    @app.callback(
        Output('sslider_status', 'children'),
        Input('telemetry_similarities', 'value')
    )
    def update_slider_label(value):
        return f"{value:.2f}"

    @app.callback(
        Output('vsslider_status', 'children'),
        Input('visual_similarities', 'value')
    )
    def update_slider_label(value):
        return f"{value:.2f}"

    @app.callback(
        Output('status', 'children'),
        [Input('filter_store', 'data'),
        Input('data_store', 'data')]
    )
    def show_status(filter_options, data):
        dfid = "" if not 'filter_id' in data else data['filter_id']
        return "done." if dfid == filter_options['filter_id'] else "loading..."

    @app.callback(
        Output('filter_store', 'data'),
        [Input('inspections_dd', 'value'),
         Input('telemetry_similarities', 'value'),
         Input('visual_similarities', 'value'),
         Input('extras_cl', 'value'),
         Input('defects_cl', 'value'),
         Input('findings_cl', 'value')]
    )
    def updateFilter(inspections, telemetry_similarities, visual_similarities, extras, defects, parts):
        title_to_id = None
        if inspections is None:
            inspections = []
        else:
            title_to_id = get_inspections()
            inspections = [title_to_id[t] for t in inspections]
        if extras is None: extras = []
        mosaics = 'mosaics' in extras
        key_frames = 'key_frames' in extras

        if defects is None: defects = []
        else: defects = [d.replace(' ','_') for d in defects]
        if parts is None: parts = []
        else: parts = [d.replace(' ','_') for d in parts]

        del(extras)
        del(title_to_id)

        filter_id = str(uuid.uuid4())

        query_filter_options = FilterOptions(**locals())
        
        print("Callback with filter data", query_filter_options)
        return dataclasses.asdict(query_filter_options)
        pass