from datetime import datetime
import inspect
import json
import os
from statistics import mean, median, stdev, variance
from typing import List
import time

import logging
logger = logging.getLogger('pipeline')

from PIL import Image
import cv2
from matplotlib.image import thumbnail

from sklearn.manifold import TSNE
from data.inspection.LiInspection import LiInspection
from data.inspection.image_node import ImageNode, MosaicNode
from data.vismodel.LiShip import LiShip

from computer_vision.LIACI_stitcher import LIACI_Stitcher
from video_input.inspection_video_input import InspectionVideo 

import data.access.frame as frame_access

from computer_vision.LIACi_classifier import LIACi_classifier
from computer_vision.LIACi_segmenter import LIACi_segmenter
from computer_vision.LIACi_detector import LIACi_detector

import computer_vision.image_quality as image_quality

import data.access.ship as ship_access
import data.access.inspection as inspection_access



def store_inspection(inspection_video: InspectionVideo, anonymize_name=True) -> None:

    logger.info(f"Started inspection analysis for video file {inspection_video.video_file}")

    statistics = {}
    start_time = time.perf_counter()
    count = 0
    try:
        if anonymize_name:
            inspection_video.anonymize_name()

        inspection_metadata = inspection_video.get_inspection_metadata()
        inspection_id = inspection_metadata.inspection_id
        imo = inspection_metadata.imo

        ship = LiShip(inspection_metadata.ship_id, imo, inspection_metadata.ship_name, inspection_metadata.ship_type, inspection_metadata.marine_traffic_url)
        inspection = LiInspection(imo, inspection_metadata.inspection_date, inspection_id)

        ship_node = ship_access.create(ship, fail_on_exists=False)
        inspection_node = inspection_access.create(inspection, fail_on_exists=True) 


        inspection_video.frame_step = 1


        classifier = LIACi_classifier()
        segmenter = LIACi_segmenter()
        detector = LIACi_detector()

        stitcher = LIACI_Stitcher()
        mosaic_node = None
        start_seconds = time.time()



        for tele, frame_array in inspection_video:
            count += 1
            logger.debug(f"Processing frame {count}")

            if frame_array is None:
                continue

            frame_number = tele['frame_index']
            frame_id = f'{inspection_id}.{frame_number}'
            frame_thumbnail_path = f'./data/imgs/frames/{frame_id}.jpg'

            neo4jnode = None

            # Do this every /*30*/ frames!
            if (count - 1) % 30 == 0:
                logger.debug(f"Frame {count} will be stored as Frame node...")
                percent = count * 100 / inspection_video.frame_count
                seconds = time.time() - start_seconds
                fps = count / seconds
                eta = (inspection_video.frame_count - count) / fps
                eta = f'{eta//3600: 3.0f}h {(eta%3600)//60: 2.0f}m {eta%60: 2.0f}s'
                print(f'\r{percent: 3.2f}% {fps: 3.1f}FPS ETA: {eta}',end='')
                statistics['fps'] = statistics.get('fps', [])+[fps]

                start = time.perf_counter()
                if not os.path.exists(frame_thumbnail_path):
                    if frame_array is not None:
                        smaller = cv2.resize(frame_array, (frame_array.shape[1] // 4, frame_array.shape[0] // 4))
                        cv2.imwrite(frame_thumbnail_path, smaller, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        
                time_store_frame = time.perf_counter()

                # do the ml stuff and store a frame node actually 🥳
                classes = classifier.classify_dict(frame_array)
                classification_done = time.perf_counter()
                detection = detector.detect(frame_array)
                objects = {d['tagName']: d['probability'] for d in detection},
                detection_done = time.perf_counter()
                segmentation = segmenter.get_coverages(frame_array)
                segmentation_done = time.perf_counter()
                quality_metric_input = cv2.resize(frame_array, (300, 169))
                uciqe = image_quality.analyse_image(quality_metric_input)
                quality_metric_done = time.perf_counter()

                image_node = ImageNode(id=frame_id, imo=imo,  framenumber=frame_number, inspection_id=inspection_id,
                    thumbnail=f'{frame_id}.jpg',
                    classes = classes,
                    segmentation = segmentation,
                    objects = objects,
                    telemetry=tele, 
                    uciqe = float(uciqe))
            
                neo4jnode = frame_access.create(image_node, inspection_node, classification_threshold=0.9)

                node_creation_done = time.perf_counter()
                logger.debug(f"Stored frame {count}")

                statistics['store frame to disk'] = statistics.get('store frame to disk', [])+[time_store_frame - start]
                statistics['classifier'] = statistics.get('classifier', [])+[classification_done - time_store_frame]
                statistics['detection'] = statistics.get('detection', [])+[detection_done - classification_done]
                statistics['segmentation'] = statistics.get('segmentation', [])+[segmentation_done - detection_done]
                statistics['quality metric'] = statistics.get('quality metric', [])+[quality_metric_done - segmentation_done]
                statistics['storage to neo4j'] = statistics.get('storage to neo4j', [])+[node_creation_done - quality_metric_done]
                    

            if mosaic_node is None:
                logger.debug(f"Mosaic was stored or aborted with last frame, creating a new one...")
                mosaic_node = frame_access.create_mosaic(MosaicNode("m" + frame_id))
                mosaic_node['start_frame'] = frame_number


            #Try to stitch, create relations to stitched image if successful
            stich_start = time.perf_counter()
            logger.debug("Running stitcher.next_frame:")
            stich_result = stitcher.next_frame(frame_array)
            stich_end = time.perf_counter()

            statistics['stitcher next frame'] = statistics.get('stitcher next frame', [])+[stich_end - stich_start]

            if stich_result is None:
                logger.debug("Stitcher returned None, evaluating Mosaic...")
                stich_store_start = time.perf_counter()
                if stitcher.number_of_frames < 180 and stitcher.get_size_increase() < 1.5:
                    logger.debug("Mosaic is bad, discarding mosaic")
                    frame_access.delete_mosaic(mosaic_node)
                    continue
                logger.debug("Mosaic is good, updating final data")
                cocos = stitcher.get_coco()
                for l in stitcher.labels:
                    mosaic_node[f'{l}_coco_size'] = cocos[l]['size']
                    mosaic_node[f'{l}_coco'] = cocos[l]['counts']

                percentages = stitcher.get_percentages()
                for l in stitcher.labels:
                    mosaic_node[f'{l}_percentage'] = float(percentages[l])

                x_dim, y_dim = stitcher.get_dimensions()
                mosaic_node['x_dim'] = x_dim
                mosaic_node['y_dim'] = y_dim
                mosaic_node['end_frame'] = frame_number
                mosaic_node['image_file'] = f'{mosaic_node["id"]}.jpg'
                mosaic_node['size_increase'] = float(stitcher.get_size_increase())
                mosaic_node['seg_image_file'] = f'{mosaic_node["id"]}_seg.jpg'

                stitcher.store_mosaic(f'./data/imgs/mosaics/{mosaic_node["id"]}.jpg')
                stitcher.store_mosaic_seg(f'./data/imgs/mosaics/{mosaic_node["id"]}_seg.jpg')

                frame_access.merge_node(mosaic_node)

                stich_store_end = time.perf_counter()

                logger.debug("Mosaic updated")
                statistics['stitcher store frame'] = statistics.get('stitcher store frame', [])+[stich_store_end - stich_store_start]
                mosaic_node = None
            else:
                if neo4jnode is not None:
                    homography_store_start = time.perf_counter()
                    frame_access.add_homography(neo4jnode, mosaic_node, stich_result)
                    homography_store_end = time.perf_counter()
                    statistics['store homography'] = statistics.get('store homography', [])+[homography_store_end - homography_store_start]
        print(f"Stored frames into neo4j graph")
    except KeyboardInterrupt:
        print("Interrupted Inspection!")


    total = time.perf_counter() - start_time

    print("statistics:")
    print(f"+-{'-'*30:30s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+")
    print(f"| {'key':30s} | {'samples':12s} | {'mean':12s} | {'median':12s} | {'stdev':12s} | {'variance':12s} |")
    print(f"+-{'-'*30:30s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+")
    for key, list in statistics.items():
        list = [l * 1000 for l in list]
        print(f"| {key:30s} | {len(list):>12d} | {mean(list):>12.3f} | {median(list):>12.3f} | {stdev(list):>12.3f} | {variance(list):>12.3f} |")
    print(f"+-{'-'*30:30s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+-{'-'*12:12s}-+")
    print()
    print(f"Total time: {total//3600}h {(total%3600)//60}m {total%60:.2f}s")
    print(f"== {count / total:.2f} frames per second ==")

    with open(f"./statistics/{inspection_id}.json", "w") as jf:
        json.dump(statistics, jf) 
            
            
