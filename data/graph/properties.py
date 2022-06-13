from typing import List, Optional
from pydantic import BaseModel


class DrawingProperties(BaseModel):
    id: str
    filename: str
    originX: int
    originY: int
    m_per_pixel_X: float 
    m_per_pixel_Y: float


class FindingProperties(BaseModel):
    id: str
    visCode: str
    description: str

#@Brian: the classes should not be fhardcoded but rather come from an external source. Usually the model provides 
# the results of theclassification
class ImageProperties(BaseModel):
    id: str
    img_filename: str
    video_id: str
    classifier_results: str
    location_x_m: float
    location_y_m: float
    location_z_m: float
    description: str


class InspectionProperties(BaseModel):
    id: str
    inspection_object: str
    date: str


class ShipProperties(BaseModel):
    id: str
    imo: str
    name: str
    type: str
    marine_traffic_url: str


class VideoProperties(BaseModel):
    id: str
    filename: str
