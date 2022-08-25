from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
@dataclass_json
@dataclass
class LiPropellerShaftArrangement():
    visCode : str = field(default='412.72', init=False)
    name : str = field(default='Propeller shaft arrangement', init=False)


@dataclass_json
@dataclass
class LiShaftSealTightness():
    visCode : str = field(default='412.723', init=False)
    name : str = field(default='Shaft seal tightness', init=False)


@dataclass_json
@dataclass
class LiShaftPropellerKeyArrangement():
    visCode : str = field(default='412.725', init=False)
    name : str = field(default='Shaft/ propeller key arrangement', init=False)
