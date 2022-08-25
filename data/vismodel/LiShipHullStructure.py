from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
@dataclass_json
@dataclass
class LiShipHullStructure():
    visCode : str = field(default='111', init=False)
    name : str = field(default='Ship Hull Structure', init=False)
