import json
from typing import List
import glob
import os

from pipeline.video_input.inspection import AnalyzedInspection


"""Class to find inspection videos with metadata.

A inspection video with metadata consists of a video file (.mp4)
with a JSON file containingall the metadata of the inspection."""
class VideoFileInput:
    def __init__(self, root_path) -> None:
        self.files = None
        self.inspections = None
        self.root_path = root_path

    def list_inspections(self) -> List:
        self.files = []
        for video_file in glob.iglob(f'{self.root_path}/**/*.mp4'):
            contex_json = video_file + '.json'

            if os.path.exists(contex_json): 
                video_file_size = os.path.getsize(video_file)
                context_file_size = os.path.getsize(contex_json)

                self.files.append({
                    'video_file': video_file, 
                    'context_file': contex_json,
                    'video_file_size': video_file_size,
                    'context_file_size': context_file_size,
                    })
                continue

        self.files.sort(key=lambda x: x['video_file'])
        return self.files 
    
    def read_inspections(self):
        if not self.files:
            _ = self.list_inspections()
        
        if self.inspections:
            for inspection in self.inspections:
                yield inspection
        
        self.inspections = []

        for inspection_file in self.files:
            video_file = inspection_file['video_file']
            context_json_file = inspection_file['context_file']

            inspection = AnalyzedInspection(video_file, context_json_file)
            self.inspections.append(inspection)
            yield inspection 

        
            
    