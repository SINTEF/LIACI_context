from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class LiSeaWaterSystem():
    visCode : str = field(default='631', init=False)
    name : str = field(default='Sea Water System', init=False)


@dataclass_json
@dataclass
class LiOpenings():
    visCode : str = field(default='631.1', init=False)
    name : str = field(default='Openings', init=False)
