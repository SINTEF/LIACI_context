from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
@dataclass_json
@dataclass
class LiManeuveringThrusterArrangement():
    visCode : str = field(default='440', init=False)
    name : str = field(default='Maneuvering thruster arrangement', init=False)
