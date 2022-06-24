from datetime import datetime
import json
import os
from typing import List
import time


from PIL import Image

import cv2
from matplotlib.image import thumbnail

from sklearn.manifold import TSNE
from data.inspection.LiInspection import LiInspection
from data.inspection.image_node import ImageNode
from data.vismodel.LiShip import LiShip
from pipeline.computer_vision.LIACI_stitcher import LIACi_stitcher
from pipeline.video_input.inspection import AnalyzedInspection, Inspection

import data.access.frame as frame_access

from pipeline.computer_vision.LIACi_classifier import LIACi_classifier
from pipeline.computer_vision.LIACi_segmenter import LIACi_segmenter
from pipeline.computer_vision.LIACi_detector import LIACi_detector


class Pipeline:
    def __init__(self) -> None:
        pass

    def store_inspection(self, inspection_data: Inspection, anonymize_name=True) -> None:

        print(f"Running pipeline for inspection {inspection_data.video_file}")
        print(f"Enabled modules:")

        ship_node, inspection_node = inspection_data.get_nodes(anonymize_name=anonymize_name)

        inspection_id = inspection_node['id']
        imo = ship_node['imo']

        inspection_data.frame_step = 1


        classifier = LIACi_classifier()
        segmenter = LIACi_segmenter()
        detector = LIACi_detector()

        stitcher = LIACi_stitcher()


        count = 0
        for frame in inspection_data:
            count += 1
            if count % 100 == 0:
                print(f"Currently analyzing frame {count}")

            if frame is None:
                continue

            tele = frame['telemetry']
            frame_number = tele['frame_index']
            frame_id = f'{inspection_id}.{frame_number}'
            frame_thumbnail_path = f'./assets/thumb/{frame_id}.jpg'


            if count % 30 == 0:
                # Store a frame to disk for visualisation
                if not os.path.exists(frame_thumbnail_path):
                    if frame['frame'] is not None:
                        smaller = cv2.resize(frame['frame'], (320, 256))
                        cv2.imwrite(frame_thumbnail_path, smaller, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        
                # do the ml stuff and store a frame node actually 🥳
                start = time.perf_counter()
                classes = classifier.classify_dict(frame)
                detection = detector.detect(frame)
                objects = {d['tagName']: d['probability'] for d in detection},
                segmentation = segmenter.get_coverages(frame)
                inftime = time.perf_counter() - start

                image_node = ImageNode(id=frame_id, imo=imo,  framenumber=frame_number, inspection_id=inspection_id,
                    video_source = inspection_data.video_file, thumbnail=f'{frame_id}.jpg',
                    classes = classes,
                    segmentation = segmentation,
                    objects = objects,
                    telemetry=tele)
            
                neo4jnode = frame_access.create(image_node, classification_threshold=0.9)

            else:
                neo4jnode = None

            #Try to stitch, create relations to stitched image if required

            could_stitch = stitcher.next_frame(frame)

            if not could_stitch:
                # Save image, create relations of frames, reset everything
                pass


        print(f"Stored {count} frames into neo4j graph, calculating similarity...")

            
            
