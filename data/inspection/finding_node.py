from datetime import date


class FindingNode(object):
    def __init__(self, id=None, visCode=None, description=None, imo=None, date=None):
        self.label = "Finding"
        self.id = id
        self.visCode = visCode
        self.description = description
        self.imo = imo
        self.date = date
