from datetime import date


class DrawingNode(object):
    def __init__(self, id=None, filename=None, imo=None, date=None):
        self.label = "Drawing"
        self.id = id
        self.filename = ""
        self.imo = imo
        self.date = date
