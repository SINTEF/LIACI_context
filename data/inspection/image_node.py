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

    label : str = dataclasses.field(default="Frame", init=False)
    id : str
    imo : str
    framenumber : str
    inspection_id : str
    thumbnail : str
    uciqe : float
    telemetry : dict = None
    classes : dict = None
    objects : dict = None
    segmentation : dict = None

    def set_neo4j_properties(self):
        for label, value in self.telemetry.items():
            setattr(self, label, value)

        for label in ['anode', 'bilge_keel', 'sea_chest_grating', 'defect', 'corrosion', 'marine_growth', 'over_board_valve', 'paint_peel', 'propeller']:
            score = 0
            if label in self.classes:
                if self.classes[label] > 0.5: score+=1 #Confidence Threshold taken from the SDD of the Liaci project documentation
                setattr(self, f"{label}_classification", self.classes[label])
            if label in self.objects:
                setattr(self, f"{label}_detection", self.objects[label])
            if label in self.segmentation:
                if self.segmentation[label] > 0.01: # At least a decent area of 1% of the image is detected to be the label. The threshold is set in the segmenter.
                    score += 1
                setattr(self, f"{label}_segmentation", self.segmentation[label])

            setattr(self, label, score)

        delattr(self, 'telemetry')
        delattr(self, 'classes')
        delattr(self, 'objects')
        delattr(self, 'segmentation')

