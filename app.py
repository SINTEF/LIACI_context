from dataclasses import replace
from platform import node
import dash
import dash_cytoscape as cyto
from dash import dcc, Input, Output, State, html
import py2neo
from py2neo import NodeMatcher, Graph, RelationshipMatcher

import networkx as nx

from data.datastore import liaci_graph, neo4j_transaction
from data.access.query import get_labels

import logging
logging.basicConfig(level=logging.WARN)

app = dash.Dash(__name__)


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

finding = ['defect', 'corrosion', 'marine_growth', 'paint_peel', 'anode', 'over_board_valve', 'propeller', 'sea_chest_grating', 'bilge_keel']

empty_elements = {
    'nodes': [
        {'data':{'label': 'Nothing to show', 'id': 3}, 'classes' : 'finding'},
    ],
    'edges': [
    ]
}

def redact_json_data(json_data, positions):
    for i, n in enumerate(json_data['elements']['nodes']):
        if 'classes' in n['data']: n['classes'] = n['data']['classes']
        elif 'imo' in n['data']: n['classes'] = 'ship'
        elif 'inspection_id' in n['data']: n['classes'] = 'inspection'
        else: n['classes'] = 'frame'
        if 'style' in n['data']: n['style'] = n['data']['style']
        pos = positions[int(n['data']['id'])]
        n['position'] = {'x': 1000 * pos[0], 'y': 1000 * pos[1]}
    for i, e in enumerate(json_data['elements']['edges']):
        if 'classes' in e['data']: e['classes'] = e['data']['classes']
    return json_data



@app.callback(
    Output(component_id='gr', component_property='elements'),
    Input(component_id='submit-btn', component_property='n_clicks'),
    State(component_id='in', component_property='value')
)
def update_graph(_, input_value):
    print(f"updating graph with value {input_value}")
    input_value = input_value.replace("paint peel", "paint_peel").replace("marine growth", "marine_growth").replace("over board valve", "over_board_valve").replace("sea chest grating", "sea_chest_grating").replace("bilge keel", "bilge_keel")
    input_value.replace(",", " ")

    d = [s for s in input_value.split() if s in finding]

    where_clause = "WHERE " + " and ".join([f'fic.{f} > 0.9' for f in d]) if d else ""

    nodes_indexes = set()
    nodes = []
    rels = []

    relcount = 0
    graph = nx.Graph()

    with neo4j_transaction() as tx:
        query = f"""MATCH (fic:Image) {where_clause} WITH fic LIMIT 200 MATCH (fic) -[r]-> (n2) RETURN fic, n2"""
        query = f"""MATCH (fic:Image) {where_clause} WITH fic LIMIT 100 MATCH (ship:Ship) -[:HAS*]-> (part) <-[:DEPICTS]- (fic:Image) -[r]-> (n2:Image) RETURN ship, part, fic, n2, r LIMIT 100"""

        query_image_nodes = f"MATCH (i:Image) WHERE {where_clause} with i order by i.id asc limit 100"



        result = tx.run(query)


        
        print("got result from database, loading result... ")
        for (ship, part, n1, n2, r) in result:
            relcount += 1
            for n in [ship, part, n1, n2]:
                if not n.identity in nodes_indexes:
                    nodes_indexes.add(n.identity)
                    graph.add_node(n.identity, label=n['name'] if 'name' in n else f'{n._labels}', image_path='./assets/thumb/' + n['thumbnail'] if 'thumbnail' in n else '', classes='frame' if 'Image' in n._labels else 'ship' if 'Ship' in n._labels else 'ship_part')
                    #nodes.append({'data':{'id': n.identity, 'label': f'{n.labels}', 'image_path': './assets/thumbnails/' + n['thumbnail'] if 'thumbnail' in n else ''}, 'classes': 'frame' if 'Image' in n._labels else 'finding'})

            is_similarity = "SIMILAR_TO" in r.types()
            is_visual_similarity = "VISUALLY_SIMILAR_TO" in r.types()
            traction = 1/20 if is_similarity else 1/100
            graph.add_edge(n1.identity, n2.identity, traction=1/50, classes='dashed' if is_similarity else 'dotted' if is_visual_similarity else '')
            graph.add_edge(ship.identity, part.identity, traction=1/500, classes='')
            graph.add_edge(n1.identity, part.identity, traction=1/100, classes='')
            #rels.append({'data':{'source': n1.identity, 'target': n2.identity}})

    if not nodes_indexes:
        return empty_elements

    print(f"loaded result, got {len(nodes_indexes)} nodes and {relcount} relationships, layouting...")
    pos = nx.spring_layout(graph, weight='traction', k=1/15, iterations=100, threshold=1e-4)
    elmts = redact_json_data(nx.cytoscape_data(graph), pos)['elements']
    print(f"done")

    return elmts



app.layout = html.Div([
    html.H1("LIACi Knowledge Graph"),
    html.Div(["Parameters:",
        dcc.Input(id='in', value=''),
        html.Button('Submit', id='submit-btn', n_clicks=0),]),
    html.Br(),
    cyto.Cytoscape(
        id='gr',
        layout={
            'name': 'preset',
            'animate': True,
            #'idealEdgeLength': 400,
            #'nodeOverlap': 20,
            #'refresh': 20,
            'fit': True,
            #'padding': 30,
            #'randomize': False,
            'componentSpacing': 1,
            'nodeRepulsion': 8000,
            #'edgeElasticity': 10,
            #'nestingFactor': 5,
            #'gravity': 80,
            #'numIter': 1000,
            #'initialTemp': 200,
            #'coolingFactor': 0.95,
            #'minTemp': 1.0
        },
        style={'width': '100%', 'height': '1000px'},
        elements={},
        stylesheet = [
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

                        "label": "data(label)",
                        "color": "black",
                        "text-opacity": 1,
                        "font-size": 18,   
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
                    'selector': '.invisible',
                    'style': {
                        'opacity': 0
                    }
                },
                {
                    'selector': '.dashed',
                    'style': {
                        'line-style': 'dashed',
                    }
                },
                {
                    'selector': '.dotted',
                    'style': {
                        'line-style': 'dotted',
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
        )
])


if __name__ == '__main__':
    app.run_server(debug=True)

