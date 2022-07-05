# Query all nodes
import math
import numpy as np
import py2neo
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data.datastore import neo4j_transaction
from img2vec_pytorch import Img2Vec
from scipy.spatial import KDTree
from PIL import Image

import data.access.frame as frame_access

def sanitize_infitinty(i):
    try:
        f = float(i)
        if math.isinf(f) or math.isnan(f):
            return 0
        else:
            return f
    except:
        return 0

def do_similarity():
    im2vec = Img2Vec(cuda=True)

    query = "MATCH (n:Image) <-[:HAS_FRAME]- (i:Inspection) RETURN n, i.id"

    py2neonodes = []
    inspection_ids = []
    id_in_inspection_list_to_id_in_node_list = {}
    id_in_node_list_to_id_in_inspectioin_list = []
    vecs = {}
    imvecs = []

    with neo4j_transaction() as tx:
        num_nodes = tx.run("MATCH (n:Image) RETURN count(n) as n")
        num_nodes = next(iter(num_nodes))['n']
        print(f"Found {num_nodes} nodes, quering all nodes...")
        result = tx.run(query)
        print("done.")
    # For each node, calculate vec
        for index, (n, inspection) in enumerate(result):
    #   Telemetry stuff
            vec = [
                sanitize_infitinty(n[field]) for field in 'Depth Heading Pitch Roll framenumber anode bilge_keel corrosion defect marine_growth over_board_valve paint_peel propeller sea_chest_grating'.split()
            ]
    #   (Image2Vec)
            img = Image.open(f"assets/thumb/{n['id']}.jpg")
            imvec = im2vec.get_vec(img, tensor=False)

            print(f"{index} / {num_nodes} ({index * 100 / num_nodes:.2f}%)\r", end="")

            py2neonodes.append(n)
            inspection_ids.append(inspection)
            if not inspection in vecs:
                vecs[inspection] = []
                id_in_inspection_list_to_id_in_node_list[inspection] = []
            id_in_inspection_list_to_id_in_node_list[inspection].append(index)
            id_in_node_list_to_id_in_inspectioin_list.append(len(vecs[inspection]))
            vecs[inspection].append(vec)
            imvecs.append(imvec)

    # TSNE
    vec_representations = {}
    for inspection, vectors in vecs.items():
        a = np.array(vectors)
        a = a / np.abs(a).max(axis=0)
        tsne = TSNE(n_components=3, learning_rate=400, verbose=1)
        vec_representations[inspection] = tsne.fit_transform(a)
    print("tsne ready")    

    a = np.array(imvecs)
    a = a / np.abs(a).max(axis=0)
    pca = PCA(n_components=15)
    imvec_representations = pca.fit_transform(a)
    print("pca ready")    

    # KD Tree

    vec_tree = {}
    for inspection, representations in vec_representations.items():
        vec_tree[inspection] = KDTree(representations)

    imvec_tree = KDTree(imvec_representations)

    # Neighbor for each node
    print("searching for neighbors and adding relaitons...")
    for i, node in enumerate(py2neonodes):
        inspection = inspection_ids[i]
        id_in_inspection_list = id_in_node_list_to_id_in_inspectioin_list[i]
        rtsne = vec_representations[inspection][id_in_inspection_list]
        rpca = imvec_representations[i]

        vec_distances, vec_neighbors = vec_tree[inspection].query(rtsne, k=5)
        imvec_distances, imvec_neighbors = imvec_tree.query(rpca, k=15)

        vec_neighbors = [py2neonodes[id_in_inspection_list_to_id_in_node_list[inspection][n]] for n in vec_neighbors]
        imvec_neighbors = [py2neonodes[n] for n in imvec_neighbors]

        for distance, neighbor in zip(vec_distances, vec_neighbors):

            if neighbor.identity != node.identity and distance < 2:
                frame_access.add_similarity(node, neighbor, float(distance), visual=False)

        for distance, neighbor in zip(imvec_distances, imvec_neighbors):
            if neighbor.identity != node.identity and distance < 2:
                frame_access.add_similarity(node, neighbor, float(distance), visual=True)

        
        print(f"{i} / {len(py2neonodes)} ({i * 100 / len(py2neonodes):.2f}%)\r", end="")
    print("Completed!")
        


        
