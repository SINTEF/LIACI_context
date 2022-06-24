import time
import json
import os
import hashlib

import random
import cv2
from cv2 import CAP_PROP_FRAME_COUNT
import numpy as np

from pipeline.computer_vision.LIACi_classifier import LIACi_classifier
from pipeline.computer_vision.LIACi_segmenter import LIACi_segmenter
from pipeline.computer_vision.LIACi_detector import LIACi_detector
from data.vismodel.LiShip import LiShip
from data.inspection.LiInspection import LiInspection
import data.access.ship as ship_access
import data.access.inspection as inspection_access

from pipeline.video_input.ass_telemetry_reader import get_telemetry_data, read_telemetry_data

class StepIterator:
    def __init__(self, object, step, get_method) -> None:
        self.object = object
        self.step = step
        self.current = 0
        self.get_method = get_method

    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            self.current += self.step
            return self.get_method(self.object, self.current - self.step)
        except:
            raise StopIteration

def random_name() -> str:
    name = random.choice("""Cambridge
Fastnet
The Olympia
Teviot
Oryx
Pegwellbay
Southwold
Fifi
Hepatica
Orontes
Hazardous
York
Helder
Abundance
Selene
Rattler
Aberfoyle
Romney
Amphion
The Badminton
Messenger
Glommen""".split())
    if random.random() > 0.5:
        name += random.choice([" I", " II", " III"])
    return name

class Inspection:
    frame_step: int
    def __init__(self) -> None:
        pass

    def get_inspection_metadata(self):
        data = {}
        get_or_ask_inspection_metadata(data, self.inspection_metadata_file)
        return data

    def get_nodes(self, anonymize_name=True, fail_on_exists=False):
        data = self.get_inspection_metadata()

        inspection_id = data['inspection_id']
        imo = data['imo']

        if anonymize_name:
            data['ship_name'] = random_name()

        ship = LiShip(data['ship_id'], imo, data['ship_name'], data['ship_type'], data['marine_traffic_url'])
        inspection = LiInspection(imo, data['inspection_date'], inspection_id)

        
        ship_node = ship_access.create(ship, fail_on_exists=fail_on_exists)

        inspection_node = inspection_access.create(inspection, fail_on_exists=fail_on_exists) 
        return ship_node, inspection_node

    def __iter__(self):
        pass

    def __next__(self):
        pass

def detid(str):
    str = hashlib.md5(str.encode('utf-8')).hexdigest()
    n = 46663
    id = 7984002041
    for c in str:
        id += ord(c) * n
    return id % 1000000



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

class VideoInspection(Inspection):
    def __init__(self, video_file, ass_file = None, metadata_file = None, frame_step=30) -> None:
        super().__init__()

        if metadata_file is None:
            metadata_file = f'{video_file}.inspection_meta.json'
        self.inspection_metadata_file = metadata_file
        if ass_file is None:
            ass_file = os.path.splitext(video_file)[0] + '.ass'
        self.ass_file = ass_file

        self.frame_step = frame_step
        self.current_frame = 0

        self.video_file = video_file
        self.cv_capture = cv2.VideoCapture(self.video_file)
        if not self.cv_capture.isOpened():
            raise IOError("Could not open video file.")
        
        self.video_length = int(self.cv_capture.get(CAP_PROP_FRAME_COUNT))

        self.telemetry = read_telemetry_data(self.ass_file, self.video_length)
        print(len(self.telemetry))

        self.classifier = LIACi_classifier()
        self.segmenter = LIACi_segmenter()
        self.detector = LIACi_detector()
    
    def __iter__(self):
        self.current_frame = 0
        self.telemetry_iterator = self.telemetry.iterrows()
        self.cv_capture.set(1, 0)
        return self
    
    def __next__(self):
        if self.current_frame > self.video_length:
            raise StopIteration()
        while self.current_frame % 30 != 0:
            _ = next(self.telemetry_iterator)
            ret, frame = self.cv_capture.read()
            self.current_frame += 1
        telemetry = next(self.telemetry_iterator)
        ret, frame = self.cv_capture.read()
        self.current_frame += 1

        if not ret:
            return None
        
        start = time.perf_counter()
        classification = self.classifier.classify_dict(frame)
        detection = self.detector.detect(frame)
        segmentation = self.segmenter.segment_unet(frame)
        inftime = time.perf_counter() - start

        
        data = {
            'frame': frame,
            'classes': classification,
            'objects': {d['tagName']: d['probability'] for d in detection},
            'segmentation': segmentation,
            'telemetry': json.loads(telemetry[1].to_json())
        }
        return data
            


class AnalyzedInspection(Inspection):
    def __init__(self, video_file, context_json_file, metadata_file = None) -> None:
        super().__init__()


        if metadata_file is None:
            metadata_file = f'{video_file}.inspection_meta.json'
        self.inspection_metadata_file = metadata_file
        self.video_file = video_file
        self.cv_capture = cv2.VideoCapture(self.video_file)
        if not self.cv_capture.isOpened():
            raise IOError("Could not open video file.")
        
        self.video_length = int(self.cv_capture.get(CAP_PROP_FRAME_COUNT))


        self.context_json_file = context_json_file
        print(f"reading {self.context_json_file}")

        self.context = json.load(open(context_json_file))
        pass


    def get_frame_array(self, frame_id):
        if not (0 < frame_id < self.video_length):
            return ValueError("Frame index out of range!")

        self.cv_capture.set(1, frame_id)
        _, frame = self.cv_capture.read()
        return frame

    def get_frame_meta(self, frame_id):
        try:
            return self.context[frame_id]
        except KeyError:
            raise ValueError("Frame index out of range.")
        
    def get_meta_iter(self, each_n_frame=1):
        return StepIterator(self, each_n_frame, 
            lambda inspection, frame_id: (inspection.get_frame_array(frame_id), inspection.get_frame_meta(frame_id)))
