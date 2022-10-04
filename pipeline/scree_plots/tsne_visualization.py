graph_stylesheet = [
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
                        'selector': '.frame',
                        'style': {
                            'background-image': 'data(image_path)',
                            "background-fit": "cover",
                            'shape' :'rectangle',
                            "opacity": 1,
                            "width": 60,
                            "height": 34,
                            "label": "data(label)",
                            "color": "#B10DC9",
                            "text-opacity": 0,
                            "font-size": 12,   
                        }
                    },
                ]

# Query all nodes
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data.access.datastore import neo4j_transaction
from img2vec_pytorch import Img2Vec
from PIL import Image, ImageOps
import dash
from dash import html, dcc
import dash_cytoscape as cyto

def sanitize_infitinty(i):
    try:
        f = float(i)
        if math.isinf(f) or math.isnan(f):
            return 0
        else:
            return f
    except:
        return 0

def get_vectors(inspection_id):
    im2vec = Img2Vec(cuda=True)
    query = f"MATCH (n:Frame) <-[:HAS_FRAME]- (i:Inspection{{id:{inspection_id}}}) RETURN n, i.id order by n.id asc"
    vecs = []
    imvecs = []
    with neo4j_transaction() as tx:
        num_nodes = tx.run("MATCH (n:Frame) RETURN count(n) as n")
        num_nodes = next(iter(num_nodes))['n']
        result = tx.run(query)
    # For each node, calculate vec
        for index, (n, inspection) in enumerate(result):
    #   Telemetry stuff
            vec = [
                #sanitize_infitinty(n[field]) for field in 'Depth.Camera Tilt.Heading.framenumber.Pitch.Roll.anode.bilge_keel.corrosion.defect.marine_growth.over_board_valve.paint_peel.propeller.sea_chest_grating'.split('.')
                sanitize_infitinty(n['Depth']),
                sanitize_infitinty(n['Camera Tilt']),
                math.sin(float(n['Heading']) * 2 * math.pi / 360),
                math.cos(float(n['Heading']) * 2 * math.pi / 360)
            ]
            vecs.append(vec)
    #   (Image2Vec)
            with Image.open(f"../data/imgs/frames/{n['id']}.jpg") as img:
                img = ImageOps.grayscale(img).convert("RGB")
                imvec = im2vec.get_vec(img, tensor=False)
                imvecs.append(imvec)
    return vecs, imvecs


def pca_to_50(vectors):
    ipca = PCA(n_components=50)
    a = np.array(vectors)

    scaler = StandardScaler()
    a = scaler.fit_transform(a)
    representations = ipca.fit_transform(a)
    return representations 

def tsne_to_2(vectors):
    tsne = TSNE(n_components=2)
    a = np.array(vectors)
    representations = tsne.fit_transform(a)
    return representations

def fuse_data(vecs, imvecs):
    if not len(vecs) == len(imvecs):
        raise ValueError("Vectors must be of same size")
    
    v = []
    for tel, im in zip(vecs, imvecs):
        l = []
        l.extend(tel)
        l.extend(im)
        v.append(l)

    return v

def get_tsne_data(inspection_id):
    vecs, imvecs = get_vectors(inspection_id)
    imvecs = pca_to_50(imvecs)
    return tsne_to_2(fuse_data(vecs, imvecs))
    

def get_inspections():
    query = "MATCH (i:Inspection) -[:HAS_INSPECTION]- (s:Ship) RETURN s.name as ship_name, i.id as inspection_id"
    with neo4j_transaction() as tx:
        result = tx.run(query)
        return {i:n for n, i in result}
        
if __name__ == "__main__":
    app = dash.Dash("App") 
    app.title = "TSNE Visualization"

    inspections = get_inspections()

    @app.callback(
        dash.Output('gr','elements'), 
        dash.Input('perplexity_slider', 'value'),
        dash.Input('step_slider', 'value'),
        dash.Input('eps_slieer', 'value')
    )
    def update_graph(perplexity, steps, epsilon):
        elmts = {}
        return elmts

    


    app.layout=html.Div(className="rootContainer", children=[
        html.Div(className="sliderContainer", children=[
            dcc.Dropdown(inspections, id='inspections_dd', multi=False),
            dcc.Slider(id="perplexity_slider", min=2, max=100, step=1, value=30),
            dcc.Slider(id="steps_slider", min=10, max=5000, step=10, value=2500),
            dcc.Slider(id="eps_slider", min=1, max=10, step=1, value=5)
        ]),
        html.Div(className="graphContainer", children=[
            cyto.Cytoscape(
                id='gr',
                layout={'name': 'preset','fit': True},
                style={'width': '100%', 'height': '1000px'}, elements={}, stylesheet = graph_stylesheet)
        ])
    ])