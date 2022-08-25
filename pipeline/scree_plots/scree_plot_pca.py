# Query all nodes
import json
import math
import numpy as np
import py2neo
import pickle
import os.path
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from data.access.datastore import neo4j_transaction
from img2vec_pytorch import Img2Vec
from scipy.spatial import KDTree
from PIL import Image, ImageOps
import matplotlib
import matplotlib.pyplot as plt

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

def scree_plot_inspections():
    im2vec = Img2Vec(cuda=True)

    query = "MATCH (n:Image) <-[:HAS_FRAME]- (i:Inspection) RETURN n, i.id order by n.id asc"

    vecs = {}
    im_vecs = {}

    imvecs_available=False
    imvec_file = "./image_bw_vectors_pickle.pyp"
    if os.path.exists(imvec_file):
        with open(imvec_file, "rb") as f:
            im_vecs = pickle.load(f)
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
                if not imvecs_available:
                    im_vecs[inspection] = []
    #   Telemetry stuff
            vec = [
                #sanitize_infitinty(n[field]) for field in 'Depth.Camera Tilt.Heading.framenumber.Pitch.Roll.anode.bilge_keel.corrosion.defect.marine_growth.over_board_valve.paint_peel.propeller.sea_chest_grating'.split('.')
                sanitize_infitinty(n['Depth']),
                sanitize_infitinty(n['Camera Tilt']),
                math.sin(float(n['Heading']) * 2 * math.pi / 360),
                math.cos(float(n['Heading']) * 2 * math.pi / 360)
            ]
            vecs[inspection].append(vec)
    #   (Image2Vec)
            if not imvecs_available:
                with Image.open(f"data/imgs/frames/{n['id']}.jpg") as img:
                    img = ImageOps.grayscale(img).convert("RGB")
                    imvec = im2vec.get_vec(img, tensor=False)
                    im_vecs[inspection].append(imvec)


            print(f"\r{index} / {num_nodes} ({index * 100 / num_nodes:.2f}%)", end="")


        print()
        print()
        print("done.")

    if not imvecs_available:
        with open(imvec_file, "wb") as f:
            pickle.dump(im_vecs, f)
    # PCA
    for i in [1]: #range(3):
        file_name = ''
        if i == 0:
            file_name = 'all_data'
        elif i == 1:
            file_name = 'only_im2vec'
        elif i == 2:
            file_name = 'only_telemetry'
        pca_stats = []
        for inspection, telvecs in vecs.items():
            vectors = []


            ipca = PCA(n_components=50)
            ipca_representations = ipca.fit_transform(im_vecs[inspection])

            for j, telvec in enumerate(telvecs):
                imvec = im_vecs[inspection][j]
                impca_vec = ipca_representations[j]
                if i == 0:
                    v = list(telvec)
                    v.extend(impca_vec)
                    vectors.append(v)
                elif i == 1:
                    vectors.append(imvec)
                elif i == 2:
                    vectors.append(telvec)

            a = np.array(vectors)

            scaler = StandardScaler()
            a = scaler.fit_transform(a)

    
            pca = PCA(n_components=min(50, len(vectors[0])))
            _ = pca.fit_transform(a)
            PC_values = np.arange(pca.n_components_) + 1
            plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=1)
            plt.plot(PC_values, np.cumsum(pca.explained_variance_ratio_), 'go-', linewidth=2)
            plt.title(f'Scree Plot')
            plt.xlabel('Principal Component')
            plt.ylabel('Proportion of Variance Explained')
            pca_stats.append({'pca_varience':[float(x) for x in pca.explained_variance_ratio_],'cumsum':[float(x) for x in np.cumsum(pca.explained_variance_ratio_)]}) 

        print("pca ready")    
        plt.savefig(f'./scree_plots/scree_plot_pca_bw')
        plt.clf()

        with open("./scree_plots/scree_plot_data_bw.json", "w") as f:
            json.dump(pca_stats, f)


    print()
    print("Completed!")
        


        
