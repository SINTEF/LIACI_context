import uuid
from datetime import date

#import LiReport


class LiInspection(object):
    def __init__(self, imo=None, date=None, id=None):
        self.label = "Inspection"
        self.id = id
        self.imo = imo
        self.date = date

