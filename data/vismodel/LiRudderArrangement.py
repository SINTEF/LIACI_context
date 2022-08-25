from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class LiRudderArrangement():
    visCode : str = field(default='421', init=False)
    name : str = field(default='Rudder arrangement', init=False)


@dataclass_json
@dataclass
class LiRudderStock():
    visCode : str = field(default='421.2', init=False)
    name : str = field(default='Rudder stock', init=False)


@dataclass_json
@dataclass
class LiRudder():
    visCode : str = field(default='421.3', init=False)
    name : str = field(default='Rudder', init=False)


@dataclass_json
@dataclass
class LiSolePiecePintles():
    visCode : str = field(default='421.4', init=False)
    name : str = field(default='Sole piece/ pintles', init=False)


@dataclass_json
@dataclass
class LiFlapBeckerRudder():
    visCode : str = field(default='421.5', init=False)
    name : str = field(default='Flap/ becker rudder', init=False)
