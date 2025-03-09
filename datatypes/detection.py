from typing import Dict


class Detection:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def to_dict(self) -> Dict:
        return {"x": self.x, "y": self.y, "z": self.z}
