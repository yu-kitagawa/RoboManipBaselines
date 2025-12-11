import numpy as np

from .RealKinovaGen3EnvBase import RealKinovaGen3EnvBase


class RealKinovaGen3DemoEnv(RealKinovaGen3EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealKinovaGen3EnvBase.__init__(
            self,
            init_qpos=np.concatenate(
                [np.deg2rad([0.0, -20.0, 180.0, -146.0, 0.0, -50.0, 90.0]), np.array([0.0])]
            ),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx
