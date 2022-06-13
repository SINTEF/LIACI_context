from datetime import datetime
import json
import os


from PIL import Image

import cv2
from matplotlib.image import thumbnail

from sklearn.manifold import TSNE
from data.inspection.LiInspection import LiInspection
from data.inspection.image_node import ImageNode
from data.vismodel.LiShip import LiShip
from pipeline.video_input.inspection import AnalyzedInspection, Inspection

import data.access.ship as ship_access
import data.access.inspection as inspection_access
import data.access.frame as frame_access
import numpy as np
from scipy.spatial import KDTree

from data import datastore

def detid(str):
    n = 46663
    id = 7984002041
    for c in str:
        id += ord(c) * n
    return id % 10000


def get_or_ask_inspection_metadata(data, inspection_metadata_file, always_ask=False):
    if os.path.exists(inspection_metadata_file):
        with open(inspection_metadata_file, "r") as f:
            for k, v in json.load(f).items():
                data[k] = v
                
    u = lambda key: data[key] if key in data else 'unknown'
    c = lambda name, i: data[name] if name in data else i if i != '' else u(name)

    for key in ['ship_name', 'imo', 'marine_traffic_url', 'ship_type', 'inspection_date']:
        if always_ask or not key in data:
            data[key] = c(key, input(f"Please provide {key} ({u(key)}): "))

    data['ship_id'] = detid(data['ship_name'])
    print(f"Ship has ID {data['ship_id']}")
    data['inspection_id'] = detid(data['imo'] + data['inspection_date'])
    print(f"Inspection has ID {data['inspection_id']}")

    with open(inspection_metadata_file, 'w') as f:
        f.write(json.dumps(data))



class Pipeline:
    def __init__(self) -> None:
        pass


    def prepare_metadata_for_preanalyzed_inspection(self, inspection_data: AnalyzedInspection) -> None:
        print("o---------------------------------o")
        print(f"File name {inspection_data.video_file}:")
        data = {}
        get_or_ask_inspection_metadata(data, f'{inspection_data.video_file}.inspection_meta.json')
        print("o---------------------------------o")


    def store_preanalyzed_inspection(self, inspection_data: AnalyzedInspection) -> None:

        print(f"Storing pre analyzed inspection {inspection_data.video_file}")

        data = {}
        get_or_ask_inspection_metadata(data, f'{inspection_data.video_file}.inspection_meta.json')

        inspection_id = data['inspection_id']
        imo = data['imo']
        ship_id = data['ship_id']
        ship_type = data['ship_type']
        marine_traffic_url = data['marine_traffic_url']
        ship_name = data['ship_name']
        inspection_date = data['inspection_date']
        #inspection_date = datetime.strptime(inspection_date, "%Y-%m-%d")

        ship = LiShip(ship_id, imo, ship_name, ship_type, marine_traffic_url)
        inspection = LiInspection(imo, inspection_date, inspection_id)

        
        ship_node = ship_access.create(ship)
        print(ship_node)

        inspection_node = inspection_access.create(inspection) 
        print(inspection_node)

        frame_ids = []
        telemetry_vectors = {}
        frame_nodes = {}


        count = 0
        for frame_array, frame_meta in inspection_data.get_meta_iter(30):
            count += 1
            if count % 100 == 0:
                print(count)

            tele = frame_meta['telemetry']
            frame_number = tele['frame_index']
            classes = frame_meta['classes']
            frame_id = f'{inspection_id}.{frame_number}'
            frame_thumbnail_path = f'./data/thumbnails/{frame_id}.jpg'

            if not os.path.exists(frame_thumbnail_path):
                if frame_array is not None:
                    smaller = cv2.resize(frame_array, (300, 169))
                    cv2.imwrite(frame_thumbnail_path, smaller, [int(cv2.IMWRITE_JPEG_QUALITY), 70])


            frame_ids.append(frame_id)

            telemetry_vectors[frame_id] = [
                float(tele['frame_index']),
                float(tele['Depth']),
                float(tele['Heading']),
                float(tele['Roll']),
                float(tele['Pitch']),
                float(tele['Camera Tilt']),
            ]
            telemetry_vectors[frame_id].extend([v for k,v in classes.items() if True])

            image_node = ImageNode(id=frame_id, imo=imo, date=tele['date_time'], framenumber=frame_number, inspection_id=inspection_id,
                video_source = inspection_data.video_file, thumbnail=frame_thumbnail_path,
                anode=              classes['anode'] ,
                corrosion=          classes['corrosion'],
                marine_growth=      classes['marine_growth'],
                over_board_valve=   classes['over_board_valve'],
                paint_peel=         classes['paint_peel'],
                propeller=          classes['propeller'],
                sea_chest_grating=  classes['sea_chest_grating'],
                bilge_keel=         classes['bildge_keel'],
                defect=             classes['defect'])
            
            neo4jnode = frame_access.create(image_node, classification_threshold=0.9)
            frame_nodes[frame_id] = neo4jnode

        print(f"Stored {count} frames into neo4j graph, calculating similarity...")

        frame_ids.sort()
        a = np.array([telemetry_vectors[k] for k in frame_ids])
        a = a / a.max(axis=0)
        #pca = PCA(n_components=3)
        tsne = TSNE(n_components=3, learning_rate=400, verbose=1)
        representations = tsne.fit_transform(a)
        print("tsne ready")
    
        kd_tree = KDTree(representations)

        for i, frame_id in enumerate(frame_ids):

            distances, neighbors = kd_tree.query(representations[i], k=5)
            for dist, neighbor in zip(distances, neighbors):
                #add relations to neighboring nodes based on tsne distance
                neighbor_index = frame_ids[neighbor]
                if frame_id != neighbor_index and dist < 2:
                    frame_access.add_similarity(frame_nodes[frame_id], frame_nodes[neighbor_index])
            
            
