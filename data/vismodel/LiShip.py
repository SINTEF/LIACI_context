from dataclasses import field, dataclass
from dataclasses_json import dataclass_json

from data.vismodel.LiShipHullStructure import *
from data.vismodel.LiSeaWaterSystem import *
from data.vismodel.LiFreshWaterSystem import *
from data.vismodel.LiMotionTrimControlArrangement import *
from data.vismodel.LiMainStructure import *
from data.vismodel.LiPropellerArrangement import *
from data.vismodel.LiPropellerShaftArrangement import *
from data.vismodel.LiRudderArrangement import *
from data.vismodel.LiPropulsionThrusterArrangement import *
from data.vismodel.LiManeuveringThrusterArrangement import *
from data.vismodel.LiAllUnderwaterAppendages import *





@dataclass_json
@dataclass
class LiShip():
    label : str = field(default="Ship", init=False)
    id : str = None
    imo : str= None
    name : str= None
    type : str= None
    marine_traffic_url : str= None
    LiShipHullStructure : LiShipHullStructure = field(default_factory=LiShipHullStructure, init=False) 
    LiPropellerArrangement : LiPropellerArrangement = field(default_factory=LiPropellerArrangement, init=False) 
    LiPropellerBladeSealingTightness : LiPropellerBladeSealingTightness = field(default_factory=LiPropellerBladeSealingTightness, init=False)
    LiPropellerShaftArrangement : LiPropellerShaftArrangement = field(default_factory=LiPropellerShaftArrangement, init=False) 
    LiShaftSealTightness : LiShaftSealTightness = field(default_factory=LiShaftSealTightness, init=False)
    LiShaftPropellerKeyArrangement : LiShaftPropellerKeyArrangement = field(default_factory=LiShaftPropellerKeyArrangement, init=False) 
    LiSeaWaterSystem : LiSeaWaterSystem = field(default_factory=LiSeaWaterSystem, init=False) 
    LiOpenings : LiOpenings = field(default_factory=LiOpenings, init=False) 
    LiFreshWaterSystem : LiFreshWaterSystem = field(default_factory=LiFreshWaterSystem, init=False) 
    LiBoxCooler : LiBoxCooler = field(default_factory=LiBoxCooler, init=False) 
    LiMotionTrimControlArrangement : LiMotionTrimControlArrangement = field(default_factory=LiMotionTrimControlArrangement, init=False) 
    LiStabilisingFins : LiStabilisingFins = field(default_factory=LiStabilisingFins, init=False) 
    LiBilgeKeels : LiBilgeKeels = field(default_factory=LiBilgeKeels, init=False) 
    LiMainStructure : LiMainStructure = field(default_factory=LiMainStructure, init=False) 
    LiCoating : LiCoating = field(default_factory=LiCoating, init=False) 
    LiAnodes : LiAnodes = field(default_factory=LiAnodes, init=False) 
    LiRudderArrangement : LiRudderArrangement = field(default_factory=LiRudderArrangement, init=False) 
    LiRudderStock : LiRudderStock = field(default_factory=LiRudderStock, init=False) 
    LiRudder : LiRudder = field(default_factory=LiRudder, init=False) 
    LiSolePiecePintles : LiSolePiecePintles = field(default_factory=LiSolePiecePintles, init=False) 
    LiFlapBeckerRudder : LiFlapBeckerRudder = field(default_factory=LiFlapBeckerRudder, init=False) 
    LiPropulsionThrusterArrangement : LiPropulsionThrusterArrangement = field(default_factory=LiPropulsionThrusterArrangement, init=False) 
    LiHydraulicOilTightness : LiHydraulicOilTightness = field(default_factory=LiHydraulicOilTightness, init=False) 
    LiManeuveringThrusterArrangement : LiManeuveringThrusterArrangement = field(default_factory=LiManeuveringThrusterArrangement, init=False) 
    LiAllUnderwaterAppendages : LiAllUnderwaterAppendages = field(default_factory=LiAllUnderwaterAppendages, init=False) 