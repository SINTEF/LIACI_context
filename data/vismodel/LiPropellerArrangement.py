from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
@dataclass_json
@dataclass
class LiPropellerArrangement():
    visCode : str = field(default='413', init=False)
    name : str = field(default='Propeller Arrangement', init=False)


@dataclass_json
@dataclass
class LiPropellerBladeSealingTightness():
    visCode : str = field(default='413.2', init=False)
    name : str = field(default='Propeller Blade Sealing Tightness', init=False)
