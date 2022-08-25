from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
@dataclass_json
@dataclass
class LiFreshWaterSystem():
    visCode : str = field(default='632', init=False)
    name : str = field(default='Fresh Water System', init=False)


@dataclass_json
@dataclass
class LiBoxCooler():
    visCode : str = field(default='632.332', init=False)
    name : str = field(default='Box Cooler', init=False)
