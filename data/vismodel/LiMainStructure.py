class LiMainStructure():
    def __init__(self, imo=None):
        self.imo = imo
        self.visCode = "100"
        self.name = "Main Structure"
        self.inspectionCheckPoints = []
        self.findings = []
        self.inspectionImages = []


class LiCoating():
    def __init__(self, imo=None):
        self.imo = imo
        self.visCode = "102.1"
        self.name = "Coating, Marine Growth and Anti Fouling"
        self.inspectionCheckPoints = []
        self.findings = []
        self.inspectionImages = []


class LiAnodes():
    def __init__(self, imo=None):
        self.imo = imo
        self.visCode = "102.2"
        self.name = "Anodes"
        self.inspectionCheckPoints = []
        self.findings = []
        self.inspectionImages = []
