from typing import List, Optional

class ImageNode(object):
    def __init__(self, id=None, imo=None, date=None, visCode=None,
                 img_filename=None, video_id=None, framenumber=None, 
                 classes=None, segmentation=None, objects=None,
                 thumbnail=None, video_source=None,
                 finding_id=None, inspection_id=None,
                 telemetry={}):
        self.label = "Image"
        self.id = id
        self.imo = imo
        self.date = date
        self.img_filename = img_filename
        self.video_id = video_id
        self.framenumber = framenumber

        for label, value in telemetry.items():
            setattr(self, label, value)

        for label in ['anode', 'bilge_keel', 'sea_chest_grating', 'defect', 'corrosion', 'marine_growth', 'over_board_valve', 'paint_peel', 'propeller']:
            score = []
            if label in classes:
                score.append(classes[label])
                setattr(self, f"{label}_classification", classes[label])
            if label in objects:
                score.append(objects[label])
                setattr(self, f"{label}_detection", objects[label])
            if label in segmentation:
                score.append(min(segmentation[label] * 200, 1.0))
                setattr(self, f"{label}_segmentation", segmentation[label])

            setattr(self, label, sum(score) / len(score))

        self.finding_id = finding_id
        self.visCode = visCode
        self.inspection_id = inspection_id
        self.video_source = video_source
        self.thumbnail = thumbnail
