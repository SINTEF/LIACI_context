import uuid
from dash import html, dcc, Input, Output
from data_manager import get_inspections, FilterOptions
import dataclasses

def title():
    return 'Controls'

def layout():
    inspections = get_inspections().keys()
    return html.Div(className='controls', children=[
        html.H3('Choose Inspection(s)'),
        dcc.Dropdown(list(inspections), value=list(inspections)[:1], id='inspections_dd', multi=True),
        html.H3('Include'),
        html.H4('Thresholds'),
        html.P('Telemetry Similarities'),
        dcc.Slider(id="telemetry_similarities", min=0, max=5,  value=1.5),
        html.P('Visual Similarities'),
        dcc.Slider(id="visual_similarities", min=0, max=5, value=2),
        html.P('Image Quality'),
        dcc.Slider(id="quality_slider", min=0, max=15, value=4),
        html.H4('Image Stitching'),
        dcc.Checklist(['mosaics'], id='extras_cl'),
        html.H4('Inspection criteria'),
        dcc.Checklist(['marine growth', 'paint peel', 'corrosion', 'defect'], id='defects_cl'),
        html.H4('Classifications'),
        dcc.Checklist(['anode', 'propeller', 'bilge keel', 'sea chest grating', 'over board valves'], id='findings_cl'),
    ])

def register_callbacks(app):
    @app.callback(
        Output('filter_store', 'data'),
        [Input('inspections_dd', 'value'),
         Input('quality_slider', 'value'),
         Input('telemetry_similarities', 'value'),
         Input('visual_similarities', 'value'),
         Input('extras_cl', 'value'),
         Input('defects_cl', 'value'),
         Input('findings_cl', 'value')]
    )
    def updateFilter(inspections, image_quality_threshold, telemetry_similarities, visual_similarities, extras, defects, parts):
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