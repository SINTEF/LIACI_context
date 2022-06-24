import json
from typing import List
import glob
import os
import hashlib

from pipeline.video_input.inspection import AnalyzedInspection, VideoInspection


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
            context_ass = os.path.splitext(video_file)[0] + ".ass"

            video_file_size = os.path.getsize(video_file)

            # if os.path.exists(contex_json): 
            #     context_file_size = os.path.getsize(contex_json)

            #     self.files.append({
            #         'context_type': 'json',
            #         'video_file': video_file, 
            #         'context_file': contex_json,
            #         'video_file_size': video_file_size,
            #         'context_file_size': context_file_size,
            #         })
            #     continue
            
            if os.path.exists(context_ass):
                context_file_size = os.path.getsize(context_ass)
                self.files.append({
                    'context_type': 'ass',
                    'video_file': video_file, 
                    'context_file': context_ass,
                    'video_file_size': video_file_size,
                    'context_file_size': context_file_size,
                    })
                continue

        self.files.sort(key=lambda x: x['video_file'])
        return self.files 

    def get_inspection(self, index):
        return(self._get_inspection(self.files[index]))

    def _get_inspection(self, inspection_file):
        video_file = inspection_file['video_file']
        context_file = inspection_file['context_file']

        inspection = None

        if inspection_file['context_type'] == 'json':
            inspection = AnalyzedInspection(video_file, context_file)
        if inspection_file['context_type'] == 'ass':
            inspection = VideoInspection(video_file, context_file)
            
        return inspection

    
    def read_inspections(self):
        if not self.files:
            _ = self.list_inspections()
        
        if self.inspections:
            for inspection in self.inspections:
                yield inspection
        
        self.inspections = []

        for inspection_file in self.files:
            inspection = self._get_inspection(inspection_file)
            self.inspections.append(inspection)
            yield inspection 

        
            
    