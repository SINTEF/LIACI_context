# Query all nodes
from collections import Counter
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

def sanitize_infitinty(i):
    try:
        f = float(i)
        if math.isinf(f) or math.isnan(f):
            return 0
        else:
            return f
    except:
        return 0

def plot_clusters():

    im2vec = Img2Vec(cuda=True)

    query = "MATCH (n:Image) <-[:HAS_FRAME]- (i:Inspection) RETURN n, i.id order by n.id asc"

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
        num_nodes = tx.run("MATCH (n:Image) RETURN count(n) as n")
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
    kmeans_clusters = {}
    for inspection, telvecs in vecs.items():
        vectors = [] 
        imvecs2 = imvecs[inspection]

        a = np.array(telvecs)
        
        scaler = StandardScaler()
        a = scaler.fit_transform(a)

        #tsne = TSNE(n_components=2, learning_rate=400, verbose=1)
        vec_representations[inspection] = []#tsne.fit_transform(a)

        a = np.array(imvecs2)
        
        scaler = StandardScaler()
        a = scaler.fit_transform(a)

        pca = PCA(n_components=50)
        imvec_representations[inspection] = pca.fit_transform(a)
        


        for telvec, imvec in zip(telvecs, imvec_representations[inspection]):
            v = list(telvec)
            v.extend(imvec)
            vectors.append(v)
        # K Means cluster finden
        # DBSCAN ??

        a = np.array(vectors)

        # scaler = StandardScaler()
        # a = scaler.fit_transform(a)
        for e in range(200, 1000, 2):
            e /= 10.0
            print(f"{e:2f} ", end="")
            for s in range(4, 9):
    
                dbscan = DBSCAN(eps = e, min_samples=s)
                dbscan_clusters[inspection] = dbscan.fit_predict(a)
                
                n_clusters = np.max(dbscan_clusters[inspection]) + 1
                counts = np.array([val for _, val in Counter(list(dbscan_clusters[inspection])).items()])

                print(f" & {n_clusters}, {np.mean(counts):.2f}, {np.std(counts):.2f}, {list(dbscan_clusters[inspection]).count(-1):.2f}", end = "")
            print()
    
        #kmeans = KMeans(8)
        #kmeans_clusters[inspection] = kmeans.fit_predict(a)
        return
