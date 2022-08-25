from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field

@dataclass_json
@dataclass
class LiPropulsionThrusterArrangement():
    visCode : str = field(default='433', init=False)
    name : str = field(default='Propulsion thruster arrangement', init=False)


@dataclass_json
@dataclass
class LiHydraulicOilTightness():
    visCode : str = field(default='433.2', init=False)
    name : str = field(default='Hydraulic oil tightness', init=False)
