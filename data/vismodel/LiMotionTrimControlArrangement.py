from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
@dataclass_json
@dataclass
class LiMotionTrimControlArrangement():
    visCode : str = field(default='460', init=False)
    name : str = field(default='Motion and Trim Control Arrangement', init=False)


@dataclass_json
@dataclass
class LiStabilisingFins():
    visCode : str = field(default='464.1', init=False)
    name : str = field(default='Stabilising Fins', init=False)


@dataclass_json
@dataclass
class LiBilgeKeels():
    visCode : str = field(default='465', init=False)
    name : str = field(default='Bilge Keels', init=False)
