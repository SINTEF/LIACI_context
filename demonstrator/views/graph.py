graph_stylesheet = [
                    {
                        'selector': '.msg',
                        'style': {
                            "border-width": 0,
                            "border-color": "black",
                            "border-opacity": 1,
                            "opacity": 1,
                            "width": 0,
                            "height": 0,

                            "label": "data(label)",
                            "color": "#000000",
                            "text-opacity": 1,
                            "font-size": 12,   
                        }
                    },
                    {
                        'selector': '.finding',
                        'style': {
                            'background-color': 'data(color)',
                            "border-width": 0,
                            "border-color": "black",
                            "border-opacity": 1,
                            "opacity": 1,
                            "width": "data(pointsize)",
                            "height": "data(pointsize)",

                            "label": "data(label)",
                            "color": "#000000",
                            "text-opacity": 1,
                            "font-size": 18,   
                        }
                    },
                    {
                        'selector': '.ship_part',
                        'style': {
                            'background-color': 'black',
                            "border-width": 0,
                            "border-color":"black",
                            "border-opacity": 1,
                            "opacity": 1,
                            "width": "20",
                            "height": "20",

                            "label": "data(label)",
                            "color": "black",
                            "text-opacity": 1,
                            "font-size": 18,   
                        }
                    },
                    {
                        'selector': '.ship',
                        'style': {
                            'background-color': '#D10F49',
                            "border-width": 2,
                            "border-color": "black",
                            "border-opacity": 1,
                            "opacity": 1,
                            "width": 30,
                            "height": 30,

                            "label": "data(id)",
                            "color": "black",
                            "text-opacity": 1,
                            "font-size": 18,   
                        }
                    },
                    {
                        'selector': '.cluster',
                        'style': {
                            'background-color': 'green',
                            "border-width": 2,
                            "border-color": "black",
                            "border-opacity": 1,
                            "opacity": 1,
                            "width": 20,
                            "height": 20,

                            "label": "data(label)",
                            "color": "#B10DC9",
                            "text-opacity": 1,
                            "font-size": 12,   
                        }
                    },
                    {
                        'selector': '.inspection',
                        'style': {
                            'background-color': 'green',
                            "border-width": 2,
                            "border-color": "purple",
                            "border-opacity": 1,
                            "opacity": 1,
                            "width": 20,
                            "height": 20,

                            "label": "data(label)",
                            "color": "#B10DC9",
                            "text-opacity": 1,
                            "font-size": 12,   
                        }
                    },
                    {
                        'selector': '.frame',
                        'style': {
                            'background-image': 'data(image_path)',
                            "background-fit": "cover",
                            'shape' :'rectangle',
                            #"border-color": "data(border_color)",
                            #'border-width':2,
                            "opacity": 1,
                            "width": 60,
                            "height": 34,
                            "label": "data(label)",
                            "color": "#B10DC9",
                            "text-opacity": 0,
                            "font-size": 12,   
                        }
                    },
                    {
                        'selector': '.mosaic',
                        'style': {
                            'background-image': 'data(image_path)',
                            "background-fit": "cover",
                            'shape' :'rectangle',
                            "border-color": "black",
                            'border-width':1,
                            "opacity": 1,
                            "width": 120,
                            "height": 68,
                            "label": "data(label)",
                            "color": "#B10DC9",
                            "text-opacity": 0,
                            "font-size": 12,   
                        }
                    },
                    {
                        'selector': '.invisible',
                        'style': {
                            'opacity': 0
                        }
                    },
                    {
                        'selector': '.inspection',
                        'style': {
                            'line-color': '#eeeeee',
                            'line-width': 1,
                            'opacity': 0
                        }
                    },
                    {
                        'selector': '.part',
                        'style': {
                            'line-color': '#eeeeee',
                            'line-width': 1
                        }
                    },
                    {
                        'selector': '.dashed',
                        'style': {
                            'line-style': 'dashed',
                            'line-color': 'black'
                        }
                    },
                    {
                        'selector': '.dotted',
                        'style': {
                            'line-style': 'dotted',
                            'line-color': 'black'
                        }
                    },
                    {
                        'selector': 'edge',
                        'style': {
                            'width': '2',
                            'curve-style': 'haystack',
                            'haystack-radius': 0,
                            'target-arrow-shape': 'triangle'
                        }
                    },
                ]
import functools
import json
import math
import random
from dash import dcc, Input, Output, State, html
import dash_cytoscape as cyto
import py2neo
from py2neo import NodeMatcher, Graph, RelationshipMatcher

import networkx as nx

from data_manager import FilterOptions, get_graph_stuff

finding = ['defect', 'corrosion', 'marine_growth', 'paint_peel', 'anode', 'over_board_valve', 'propeller', 'sea_chest_grating', 'bilge_keel']

empty_elements = {
    'nodes': [
        {'data':{'label': 'Nothing to show', 'id': 3}, 'classes' : 'msg'},
    ],
    'edges': [
    ]
}
loading_elements = {
    'nodes': [
        {'data':{'label': 'Loading...', 'id': 3}, 'classes' : 'msg'},
    ],
    'edges': [
    ]
}

"""
elmts have the structure {'nodes': [], 'edges': []}
nodes: List of nodes:
{   
    'data':       {arbitrary data, no meaning for graph unless used in layout, eg. size}, 
    'classes':    "classes, comma seperated i guess", 
    'position':   {'x': ..., 'y': ...}
}

edges: List of edges:
{
    'data': {
        'source' : <index of node in nodes List>
        'target' : <index of node in nodes List>
        + optional arbitrary data used in layout
    }
}
"""

def redact_json_data(json_data, positions):
    for i, n in enumerate(json_data['elements']['nodes']):
        n['data']['id'] = n['data']['value'] 
        if 'classes' in n['data']: n['classes'] = n['data']['classes']
        elif n['data']['id'].startswith('s_'): 
            n['classes'] = 'ship'
            n['data']['label'] = n['data']['name']
            
        elif n['data']['id'].startswith('p_s_'): 
            n['classes'] = 'ship_part'
            n['data']['label'] = n['data']['name']
        elif n['data']['id'].startswith('in_'):
            n['classes'] = 'inspection'
            n['data']['label'] = f"Inspection on {n['data']['date']}"
        elif n['data']['id'].startswith('c'):
            n['classes'] = 'cluster'
            n['data']['label'] = f"Cluster {n['data']['id'].split('.')[-1]}"
        elif n['data']['id'].startswith('m_'): 
            n['classes'] = 'mosaic'
            n['data']['image_path'] = 'assets/imgs/mosaics/' + n['data']['seg_image_file']
        elif n['data']['id'].startswith('im_'): 
            n['classes'] = 'frame'
            n['data']['image_path'] = 'assets/imgs/frames/' + n['data']['thumbnail']
            n['data']['label'] = f"{n['data']['frame_index']}"

        if 'style' in n['data']: n['style'] = n['data']['style']
        try:
            #pos = n['data']['pca'].split(',')
            #pos = [float(p) for p in pos]
            pos = positions[n['data']['id']]
            n['position'] = {'x': 1 * pos[0], 'y': 1 * pos[1]}
        except:
            n['position'] = {'x': 0, 'y': 0}

    for i, e in enumerate(json_data['elements']['edges']):
        if 'classes' in e['data']: e['classes'] = e['data']['classes']
        e['data']['weight'] = e['data'].get('weight', 0.01)
    return json_data


legend_colors_segmenter = {
    'anode': (0, 255, 255),
    'bilge_keel': (255, 165, 0),
    'corrosion': (255, 255, 0),
    'defect': (255, 192, 203),
    'marine_growth': (0, 128, 0),
    'over_board_valve': (64, 224, 208),
    'paint_peel': (255, 0, 0),
    'propeller': (128, 0, 128),
    'sea_chest_grating': (255, 255, 255),
    'ship_hull': (0, 0, 255),
}

def layout():
    return html.Div(className="two_columns", children=[
        html.Div(className="graphContainer", children=[
        html.H1("Graph View"),
        dcc.Loading(
            id="loading_gr",
            type="default",
            children=[
                cyto.Cytoscape(
                    id='gr',
                    layout={
                        'name': 'preset',
                        'animate': False,
                        'fit': True,
                    },
                    style={'width': '100%', 'height': '1000px'},
                    elements={},
                    stylesheet = graph_stylesheet) 
            ])
        ]),
        html.Div(className="nodeInfo", children=[
           html.H1("Node Info"),
           html.P("Select node to display node information",id="nodeInfo")
        ])
    ])

def title():
    return 'Graph View'

def register_callback(app):
    @app.callback(
        Output(component_id='gr', component_property='elements'),
        Input(component_id='filter_store', component_property='data'),
    )
    def update_graph(filter_options):
        filter_options = FilterOptions(**filter_options)

        nodes, similar_to, visually_similar_to, in_inspection, in_mosaic, in_cluster, shows_part, part_of_ship = get_graph_stuff(filter_options)

        graph = nx.Graph()
        if len(nodes) == 0:
            return empty_elements


        node_tuples = []
        node_tuples = [(key, value) for key, value in nodes.items()]
        node_keys = {k for k,_ in node_tuples}

        graph.add_nodes_from(node_tuples)

        relcount = 0

        replace_with = set()

        SIMILAR_EDGE_WEIGHT = 0.6
        CLUSTER_EDGE_WEIGHT = 0.6
        ININSPE_EDGE_WEIGHT = 0.1
        PARTOFS_EDGE_WEIGHT = 0.1
        SHOWSPR_EDGE_WEIGHT = 0.1
        
        for n1, n2 in in_inspection:
            if not n1 in node_keys or not n2 in node_keys: continue
            relcount += 1
            graph.add_edge(n1, n2, classes="inspection", weight=ININSPE_EDGE_WEIGHT)
        for n1, n2 in similar_to:
            if not n1 in node_keys or not n2 in node_keys: continue
            relcount += 1
            graph.add_edge(n1, n2, classes="dashed", weight=SIMILAR_EDGE_WEIGHT)
        for n1, n2 in visually_similar_to:
            if not n1 in node_keys or not n2 in node_keys: continue
            relcount += 1
            graph.add_edge(n1, n2, classes="dotted", weight=SIMILAR_EDGE_WEIGHT)
        for n1, n2 in in_mosaic:
            if not n1 in node_keys or not n2 in node_keys: continue
            relcount += 1
            replace_with.add((n1, n2))
            graph.add_edge(n1, n2)

        for n1, n2 in in_cluster:
            if not n1 in node_keys or not n2 in node_keys: continue
            relcount += 1
            graph.add_edge(n1, n2, weight=CLUSTER_EDGE_WEIGHT)

        for n1, n2 in part_of_ship:
            if not n1 in node_keys or not n2 in node_keys: continue
            relcount += 1
            graph.add_edge(n1, n2, weight=PARTOFS_EDGE_WEIGHT)
        for n1, n2 in shows_part:
            if not n1 in node_keys or not n2 in node_keys: continue
            relcount += 1
            graph.add_edge(n1, n2, weight=SHOWSPR_EDGE_WEIGHT, classes="part")

        for image, mosaic in replace_with:
            for n1, n2, data in graph.edges(image, data=True):
                if n1 == image: n1 = mosaic
                if n2 == image: n2 = mosaic
                graph.add_edge(n1, n2, **data)
            graph.remove_node(image)

        pos = nx.spring_layout(graph, scale=1000, k=10/math.sqrt(graph.order() + 1), iterations=300, seed=84365)
        elmts = redact_json_data(nx.cytoscape_data(graph), pos)['elements']

        return elmts

    @app.callback(
        Output('nodeInfo', 'children'),
        Input('gr', 'tapNodeData'))    
    def updateNodeInfo(node_data):
        if node_data is None:
            return "Select node to display data."
        children = [html.Table(children=[
                html.Tr([
                    html.Th("key"),
                    html.Th("value")
                ])
            ] + [
                html.Tr([
                    html.Td(f"{key}"),
                    html.Td(f"{value}")
                ])
            for key, value in node_data.items() if not key.endswith("coco")]
        )] 

        if 'thumbnail' in node_data:
            children = [html.Img(style={'width':'100%'}, src=f'/assets/imgs/frames/{node_data["thumbnail"]}')] + children

        if 'seg_image_file' in node_data:
            children = [html.Img(style={'width':'100%'}, src=f'/assets/imgs/mosaics/{node_data["seg_image_file"]}')] + children
        return html.Div(children=children)