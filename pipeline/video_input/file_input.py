from dataclasses import dataclass
from typing import List
import glob
import os

from video_input.inspection_video_input import InspectionVideo, InspectionVideoFile



"""Class to find inspection videos with metadata.

A inspection video with metadata consists of a video file (.mp4)
with a JSON file containingall the metadata of the inspection."""
class VideoFileFinder:
    files:List[InspectionVideoFile]
    inspections:List[InspectionVideo]
    root_path:str
    def __init__(self, root_path : str) -> None:
        self.files = None
        self.inspections = None
        self.root_path = root_path

    def list_inspections(self) -> List[InspectionVideoFile]:
        self.files = []
        possible_video_files = []
        if os.path.isfile(self.root_path) and self.root_path.endswith('.mp4'):
            possible_video_files = [self.root_path]
        elif os.path.isdir(self.root_path):
            possible_video_files = glob.iglob(f'{self.root_path}/**/*.mp4')
        else:
            raise ValueError(f"No such file or directory: {self.root_path}")

        for video_file in possible_video_files:
            context_ass = os.path.splitext(video_file)[0] + ".ass"

            if os.path.exists(context_ass):
                context_file_size = os.path.getsize(context_ass)
                video_file_size = os.path.getsize(video_file)
                self.files.append(InspectionVideoFile(
                        context_file=context_ass,
                        video_file=video_file,
                        video_file_size=video_file_size,
                        context_file_size=context_file_size,
                        context_type='ass')
                    )

        self.files.sort(key=lambda x: x.video_file)
        return self.files 


    def get_inspection(self, inspection_file: InspectionVideoFile, ask_for_metadata=False) -> InspectionVideo:
        video_file = inspection_file.video_file
        context_file = inspection_file.context_file

        inspection = None

        if inspection_file.context_type == 'ass':
            inspection = InspectionVideo(video_file, context_file, ask_for_metadata=ask_for_metadata)
            
        return inspection

    
    def read_inspections(self, selection=None) -> InspectionVideo:
        if not self.files:
            _ = self.list_inspections()

        selected_inspections = selection if selection else self.files
        
        if self.inspections:
            for inspection in self.inspections:
                yield inspection
        
        self.inspections = []

        for inspection_file in self.files:
            inspection = self.get_inspection(inspection_file)
            self.inspections.append(inspection)
            yield inspection 

        
            
    