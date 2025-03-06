import random
import math
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
    def __init__(
        self,
        id: int,
        x: float,
        y: float,
        z: float,
        start_frame: int,
        end_frame: int,
        mod_min: float = 0.10,
        mod_max: float = 0.40,
    ):
        self.id = id
        self.start_x = x
        self.start_y = y
        self.start_z = z
        self.start_frame = start_frame
        self.end_frame = end_frame

        self.func_x = random.choice([math.sin, math.cos, lambda x: 0.20 * x])
        self.func_y = random.choice([math.sin, math.cos, lambda y: 0.20 * y])
        self.func_z = random.choice([math.sin, math.cos, lambda z: 0.20 * z])

        self.mod_x = random.uniform(mod_min, mod_max)
        self.mod_y = random.uniform(mod_min, mod_max)
        self.mod_z = random.uniform(mod_min, mod_max)

        self.track = []

    def add(self, frame: int, x: float, y: float, z: float):
        self.track.append(Reference(self.id, x, y, z))

    def generate(self) -> Dict[int, Reference]:
        positions = {}
        x, y, z = self.start_x, self.start_y, self.start_z

        for frame in range(self.start_frame, self.end_frame + 1):
            positions[frame] = Reference(self.id, x, y, z)
            x += self.func_x(self.mod_x * (frame - self.start_frame))
            y += self.func_y(self.mod_y * (frame - self.start_frame))
            z += self.func_z(self.mod_z * (frame - self.start_frame))

        return positions
