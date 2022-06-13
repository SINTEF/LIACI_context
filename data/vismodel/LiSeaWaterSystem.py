class LiSeaWaterSystem():
    def __init__(self, imo=None):
        self.imo = imo
        self.visCode = "631"
        self.name = "Sea Water System"
        self.inspectionCheckPoints = []
        self.findings = []
        self.inspectionImages = []


class LiOpenings():
    def __init__(self, imo=None):
        self.imo = imo
        self.visCode = "631.1"
        self.name = "Openings"
        self.seaChests = []
        self.inlets = []
        self.outlets = []
        self.inspectionCheckPoints = []
        self.findings = []
        self.inspectionImages = []
