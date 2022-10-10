from dataclasses import dataclass, field, fields
import json
import os
import hashlib

import random
from typing import Any
import cv2
from cv2 import CAP_PROP_FRAME_COUNT
from cv2 import VideoCapture
from dataclasses_json import dataclass_json

from video_input.ass_telemetry_reader import read_telemetry_data

@dataclass
class InspectionVideoFile:
    context_type: str
    video_file: str
    context_file: str
    video_file_size: int
    context_file_size: int

def detid(str) -> int:
    str = hashlib.md5(str.encode('utf-8')).hexdigest()
    n = 46663
    id = 7984002041
    for c in str:
        id += ord(c) * n
    return id % 1000000

@dataclass_json
@dataclass
class InspectionMetadata:
    ship_name : str = field(default="unknown")
    imo : str = field(default="unknown") 
    marine_traffic_url : str = field(default="unknown")
    ship_type : str = field(default="unknown")
    inspection_date: str = field(default="unknown")
    ship_id = 0
    inspection_id = 0

    @staticmethod
    def read_or_ask(path, always_ask = False) -> 'InspectionMetadata':
        data = InspectionMetadata()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = InspectionMetadata.from_json(f.read())
                
        #lambda to keep the original data on empty input
        c = lambda obj, key, value: value if value else getattr(obj, key)

        for key, value in data.to_dict().items():
            if 'unknown' ==  value or always_ask:
                setattr(data, key, c(data, key, input(f"Please provide {key} ({value}): ")))

        data.ship_id = detid(data.ship_name)
        print(f"Ship has ID {data.ship_id}")
        data.inspection_id = detid(data.imo + data.inspection_date)
        print(f"Inspection has ID {data.inspection_id}")

        with open(path, 'w') as f:
            f.write(data.to_json())
        return data
        

def random_name() -> str:
    name = random.choice("""Cambridge Fastnet The Olympia Teviot Oryx Pegwellbay Southwold Fifi Hepatica Orontes Hazardous York Helder Abundance Selene Rattler Aberfoyle Romney Amphion The Badminton Messenger Glommen""".split())
    if random.random() > 0.5:
        name += random.choice([" I", " II", " III"])
    return name


""" Inspection
"""
class InspectionVideo:
    frame_step: int
    frame_count: int
    video_file:str
    inspection_metadata_file:str 
    ass_file:str
    cv_capture:VideoCapture
    video_length:int
    telemetry:Any
    metadata:InspectionMetadata
    
    def __init__(self, video_file, ass_file = None, metadata_file = None, ask_for_metadata=False, frame_step=30) -> None:

        if metadata_file is None:
            metadata_file = f'{video_file}.inspection_meta.json'
        self.inspection_metadata_file = metadata_file
        if ass_file is None:
            ass_file = os.path.splitext(video_file)[0] + '.ass'
        self.ass_file = ass_file

        print(f"Loading metadata for {video_file}", flush=True)
        self.metadata = InspectionMetadata.read_or_ask(self.inspection_metadata_file, always_ask=ask_for_metadata)

        self.frame_step = frame_step
        self.current_frame = 0

        self.video_file = video_file
        self.cv_capture = cv2.VideoCapture(self.video_file)
        if not self.cv_capture.isOpened():
            raise IOError("Could not open video file.")
        
        self.video_length = self.frame_count = int(self.cv_capture.get(CAP_PROP_FRAME_COUNT))

        self.telemetry = read_telemetry_data(self.ass_file, self.video_length)
        

    def get_inspection_metadata(self):
        return self.metadata

    def anonymize_name(self):
        self.metadata.ship_name = random_name()

    
    def __iter__(self):
        self.current_frame = 0
        self.telemetry_iterator = self.telemetry.iterrows()
        self.cv_capture.set(1, 0)
        return self
    
    def __next__(self):
        if self.current_frame > self.video_length:
            raise StopIteration()
        while self.current_frame % self.frame_step != 0:
            _ = next(self.telemetry_iterator)
            ret, frame = self.cv_capture.read()
            self.current_frame += 1
        telemetry = next(self.telemetry_iterator)
        ret, frame = self.cv_capture.read()
        self.current_frame += 1

        if not ret:
            return None, None
        
        return json.loads(telemetry[1].to_json()), frame
            


