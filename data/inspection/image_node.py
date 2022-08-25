import dataclasses
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Optional

@dataclass_json
@dataclass
class MosaicNode(object):
    label : str = dataclasses.field(default='Mosaic', init=False)
    id: str
    

@dataclass_json
@dataclass
class ImageNode(object):

    label : str = dataclasses.field(default="Image", init=False)
    id : str
    imo : str
    framenumber : str
    inspection_id : str
    video_source : str
    thumbnail : str
    uiqm : float
    uciqe : float
    telemetry : dict = None
    classes : dict = None
    objects : dict = None
    segmentation : dict = None

    def set_neo4j_properties(self):
        for label, value in self.telemetry.items():
            setattr(self, label, value)

        for label in ['anode', 'bilge_keel', 'sea_chest_grating', 'defect', 'corrosion', 'marine_growth', 'over_board_valve', 'paint_peel', 'propeller']:
            score = []
            if label in self.classes:
                score.append(self.classes[label])
                setattr(self, f"{label}_classification", self.classes[label])
            if label in self.objects:
                score.append(self.objects[label])
                setattr(self, f"{label}_detection", self.objects[label])
            if label in self.segmentation:
                score.append(min(self.segmentation[label] * 200, 1.0))
                setattr(self, f"{label}_segmentation", self.segmentation[label])

            setattr(self, label, sum(score) / len(score))

        delattr(self, 'telemetry')
        delattr(self, 'classes')
        delattr(self, 'objects')
        delattr(self, 'segmentation')

