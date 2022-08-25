from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dataclasses import field
@dataclass_json
@dataclass
class LiMainStructure():
    visCode : str = field(default='100', init=False)
    name : str = field(default='Main Structure', init=False)


@dataclass_json
@dataclass
class LiCoating():
    visCode : str = field(default='102.1', init=False)
    name : str = field(default='Coating, Marine Growth and Anti Fouling', init=False)


@dataclass_json
@dataclass
class LiAnodes():
    visCode : str = field(default='102.2', init=False)
    name : str = field(default='Anodes', init=False)
