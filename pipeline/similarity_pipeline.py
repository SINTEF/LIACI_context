# Query all nodes
import json
import math
import os
import numpy as np
import py2neo
import pickle
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data.access.datastore import neo4j_transaction
from img2vec_pytorch import Img2Vec
from scipy.spatial import KDTree
from PIL import Image

import data.access.frame as frame_access
import data.access.cluster as cluster_access

def sanitize_infitinty(i):
    try:
        f = float(i)
        if math.isinf(f) or math.isnan(f):
            return 0
        else:
            return f
    except:
        return 0

def delete_all_similarities(inspection_filter = None):
    with neo4j_transaction() as tx:
        if inspection_filter is not None:
            tx.run(f"MATCH (c:Cluster) <-[:IN_CLUSTER]- (n:Frame) <-[:HAS_FRAME]- (i:Inspection) WHERE i.id in [{','.join(inspection_filter)}] DETACH DELETE c")
            tx.run(f"MATCH (i2:Frame) <-[r]- (n:Frame) <-[:HAS_FRAME]- (i:Inspection) WHERE i.id in [{','.join(inspection_filter)}] DELETE r")
        else:
            tx.run(f"MATCH () <-[r:SIMILAR_TO]- () DELETE r")
            tx.run(f"MATCH () <-[r:VISUALLY_SIMILAR_TO]- () DELETE r")
            tx.run(f"MATCH (c:Cluster) DETACH DELETE r")

def do_similarity(inspection_filter = None):

    im2vec = Img2Vec(cuda=True)
    if inspection_filter is not None:
        query = f"MATCH (n:Frame) <-[:HAS_FRAME]- (i:Inspection) WHERE i.id in [{','.join(inspection_filter)}] RETURN n, i.id order by n.id asc"
        query_number = f"MATCH (n:Frame) <-[:HAS_FRAME]- (i:Inspection) WHERE i.id in [{','.join(inspection_filter)}] return count(n) as n"
    else:
        query = f"MATCH (n:Frame) <-[:HAS_FRAME]- (i:Inspection) RETURN n, i.id order by n.id asc"
        query_number = f"MATCH (n:Frame) <-[:HAS_FRAME]- (i:Inspection) RETURN count(n) as n"

    py2neonodes = [] #Py2Neo Node instances. They can be merged to neo4j again.
    inspection_ids = [] #Inspection ID for each node instance, has same length as py2neonodes
    id_in_inspection_list_to_id_in_node_list = {} #Dictionary for each inspection ID: List of ids in py2neonodes array for the images in the inInspection ID for each node instance, has same length as py2neonodes
    id_in_node_list_to_id_in_inspectioin_list = []
    vecs = {}
    imvecs = {}
    
    imvecs_available=False
    imvec_file = "./image_bw_vectors_pickle.pyp"
    if os.path.exists(imvec_file):
        with open(imvec_file, "rb") as f:
            imvecs = pickle.load(f)
        imvecs_available = True

    with neo4j_transaction() as tx:
        num_nodes = tx.run(query_number)
        num_nodes = next(iter(num_nodes))['n']
        print(f"Found {num_nodes} nodes, quering all nodes...")
        result = tx.run(query)
        print("done.")
        print()
        print()
    # For each node, calculate vec
        for index, (n, inspection) in enumerate(result):
            if not inspection in vecs:
                vecs[inspection] = []
                id_in_inspection_list_to_id_in_node_list[inspection] = []
                if not imvecs_available:
                    imvecs[inspection] = []
            py2neonodes.append(n)
            inspection_ids.append(inspection)
            id_in_inspection_list_to_id_in_node_list[inspection].append(index)
            id_in_node_list_to_id_in_inspectioin_list.append(len(vecs[inspection]))

    #   Telemetry stuff
            vec = [
                #sanitize_infitinty(n[field]) for field in 'Depth.Heading.Camera Tilt.Pitch.Roll.framenumber.anode.bilge_keel.corrosion.defect.marine_growth.over_board_valve.paint_peel.propeller.sea_chest_grating'.split('.')
                #sanitize_infitinty(n[field]) for field in 'Depth.Heading.Camera Tilt.framenumber'.split('.')
                sanitize_infitinty(n['Depth']),
                math.sin(float(n['Heading']) * 2 * math.pi / 360),
                math.cos(float(n['Heading']) * 2 * math.pi / 360),
                sanitize_infitinty(n['Camera Tilt']),
                sanitize_infitinty(n['framenumber']),
            ]
    #   (Image2Vec)
            if not imvecs_available:
                image_file = f"data/imgs/frames/{n['id']}.jpg"
            
                with Image.open(image_file) as img:
                    imvec = im2vec.get_vec(img, tensor=False)
                    imvecs[inspection].append(imvec)

            vecs[inspection].append(vec)
            print(f"\r{index} / {num_nodes} ({index * 100 / num_nodes:.2f}%)", end="")

        print()
        print()
        print("done.")

    if not imvecs_available:
        with open(imvec_file, "wb") as f:
            pickle.dump(imvecs, f)

    # TSNE / PCA
    vec_representations = {}
    imvec_representations = {}
    dbscan_clusters = {}
    for inspection, telvecs in vecs.items():
        vectors = [] 
        imvecs2 = imvecs[inspection]

        a = np.array(telvecs)
        
        scaler = StandardScaler()
        a = scaler.fit_transform(a)

        tsne = TSNE(n_components=2, learning_rate=400, verbose=1)
        vec_representations[inspection] = tsne.fit_transform(a)

        a = np.array(imvecs2)
        
        scaler = StandardScaler()
        a = scaler.fit_transform(a)

        pca = PCA(n_components=50)
        imvec_representations[inspection] = pca.fit_transform(a)
        


        for telvec, imvec in zip(telvecs, imvec_representations[inspection]):
            v = list(telvec)
            v.extend(imvec)
            vectors.append(v)

        # DBSCAN 

        a = np.array(vectors)

        # scaler = StandardScaler()
        # a = scaler.fit_transform(a)
    
        dbscan = DBSCAN(eps = 61.4, min_samples=5)
        dbscan_clusters[inspection] = dbscan.fit_predict(a)
        print(f"found {np.max(dbscan_clusters[inspection]) + 1} clusters for in spection {inspection}")
    

    vec_tree = {}
    imvec_tree = {}
    for inspection, representations in vec_representations.items():
        vec_tree[inspection] = KDTree(representations)
        imvec_tree[inspection] = KDTree(imvec_representations[inspection])

    # Neighbor for each node
    print("searching for neighbors and adding relaitons...")
    print()
    print()
    for i, node in enumerate(py2neonodes):
        print(f"\r{i} / {len(py2neonodes)} ({i * 100 / len(py2neonodes):.2f}%)", end="")
        inspection = inspection_ids[i]
        id_in_inspection_list = id_in_node_list_to_id_in_inspectioin_list[i]
        rtsne = vec_representations[inspection][id_in_inspection_list]
        rpca = imvec_representations[inspection][id_in_inspection_list]
        imvec = imvecs[inspection][id_in_inspection_list]

        dbscan_cluster = dbscan_clusters[inspection][id_in_inspection_list]

        cluster_id = f"c{inspection}.{dbscan_cluster}"
        cluster_access.create_or_attach(node, cluster_id, int(dbscan_cluster))

        vec_distances, vec_neighbors = vec_tree[inspection].query(rtsne, k=5)
        imvec_distances, imvec_neighbors = imvec_tree[inspection].query(rpca, k=5)

        vec_neighbors = [py2neonodes[id_in_inspection_list_to_id_in_node_list[inspection][n]] for n in vec_neighbors]
        imvec_neighbors = [py2neonodes[id_in_inspection_list_to_id_in_node_list[inspection][n]] for n in imvec_neighbors]



        for distance, neighbor in zip(vec_distances, vec_neighbors):

            if neighbor.identity != node.identity:# and distance < 1.5:
                frame_access.add_similarity(node, neighbor, float(distance), visual=False)

        for distance, neighbor in zip(imvec_distances, imvec_neighbors):

            if neighbor.identity != node.identity:# and distance < 3:
                frame_access.add_similarity(node, neighbor, float(distance), visual=True)


        
    
    print()
    print()
    print()
    print()
    print("Completed!")
        


        
