import json

import cv2

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

class Inspection:
    def __init__(self) -> None:
        pass

    def get_frame_iter(self, each_n_frame=1):
        return StepIterator(self, each_n_frame, lambda inspection, frame_id: inspection.get_frame_array(frame_id))

    def get_frame_array(self, frame_id):
        pass


class AnalyzedInspection(Inspection):
    def __init__(self, video_file, context_json_file) -> None:
        super().__init__()


        self.video_file = video_file
        self.cv_capture = cv2.VideoCapture(self.video_file)
        if not self.cv_capture.isOpened():
            raise IOError("Could not open video file.")
        


        self.context_json_file = context_json_file
        print(f"reading {self.context_json_file}")

        self.context = json.load(open(context_json_file))
        pass

    def get_frame_array(self, frame_id):
        self.cv_capture.set(1, frame_id)
        _, frame = self.cv_capture.read()
        if frame is None:
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
