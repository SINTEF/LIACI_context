from dataclasses import dataclass
from dataclasses import field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class ClusterNode(object):
    label : str = field(default="Cluster", init=False)
    id: str
    
    