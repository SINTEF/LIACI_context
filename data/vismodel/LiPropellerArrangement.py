class LiPropellerArrangement():
    def __init__(self, imo=None):
        self.imo = imo
        self.visCode = "413"
        self.name = "Propeller Arrangement"
        # self.inspectionCheckPoint = ["Propeller blades", "Propeller boss", "Propeller shaft external part", "Propeller nut: securing", "Propeller hub coupling bolts or nuts: securing of the protective arrangement", "Propeller blade bolts: securing"]
        # self.tags = ["Blade bolts", "Nut", "Boss", "Hub coupling", "Marine growth", "Blades", "Shaft external part", "Other"]
        self.inspectionCheckPoints = []
        self.findings = []
        self.inspectionImages = []


class LiPropellerBladeSealingTightness():
    def __init__(self, imo=None):
        self.imo = imo
        self.visCode = "413.2"
        self.name = "Propeller Blade Sealing Tightness"
        # self.inspectionCheckPoint = ["Propeller shaft external sealing: examine for tightness", "Propeller blade sealing: examine for tightness"]
        # self.tags = ["Shaft external sealing", "Blade sealing"]
        self.inspectionCheckPoints = []
        self.findings = []
        self.inspectionImages = []
