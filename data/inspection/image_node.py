from typing import List, Optional

class ImageNode(object):
    def __init__(self, id=None, imo=None, date=None, visCode=None,
                 img_filename=None, video_id=None, framenumber=None, 
                 anode=None, corrosion=None, crack=None, marine_growth=None, over_board_valve=None, paint_peel=None, propeller=None, sea_chest_grating=None, defect=None, bilge_keel=None,
                 thumbnail=None, video_source=None,
                 finding_id=None, inspection_id=None):
        self.label = "Image"
        self.id = id
        self.imo = imo
        self.date = date
        self.img_filename = img_filename
        self.video_id = video_id
        self.framenumber = framenumber
        self.anode = anode
        self.bilge_keel = bilge_keel
        self.sea_chest_grating = sea_chest_grating
        self.defect = defect
        self.corrosion = corrosion
        self.marine_growth = marine_growth
        self.over_board_valve = over_board_valve
        self.paint_peel = paint_peel
        self.propeller = propeller
        self.crack = crack
        self.finding_id = finding_id
        self.visCode = visCode
        self.inspection_id = inspection_id
        self.video_source = video_source
        self.thumbnail = thumbnail
