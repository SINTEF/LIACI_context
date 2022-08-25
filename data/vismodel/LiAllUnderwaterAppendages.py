from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
@dataclass_json
@dataclass
class LiAllUnderwaterAppendages():
    visCode : str = field(default='173.1', init=False)
    name : str = field(default='All other underwater appendages', init=False)
