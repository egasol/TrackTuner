from typing import Dict, List


class Reference:
    def __init__(self, id: int, x: float, y: float, z: float) -> None:
        self.id = id
        self.x = x
        self.y = y
        self.z = z

    def to_dict(self) -> Dict:
        return {"id": self.id, "x": self.x, "y": self.y, "z": self.z}


class ReferenceTrack:
    def __init__(self):
        self.track: List[Reference] = []
