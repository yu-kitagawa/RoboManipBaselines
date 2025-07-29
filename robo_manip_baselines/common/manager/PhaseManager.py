import cv2
import numpy as np

from ..utils.MiscUtils import remove_suffix


class PhaseManager:
    """Phase manager."""

    def __init__(self, phase_order):
        self.phase_order = phase_order

        # Assuming that reset() is explicitly called after construct,
        # reset() is not called here to avoid duplicate initialization of the initial state.
        # self.reset()

    def reset(self):
        self._set_phase(0)

    def pre_update(self):
        self.phase.pre_update()

    def post_update(self):
        self.phase.post_update()

    def check_transition(self):
        if self.phase.check_transition():
            self._set_phase(self.phase_idx + 1)

    def _set_phase(self, phase_idx):
        if not (0 <= phase_idx < len(self.phase_order)):
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid phase index: {phase_idx}"
            )

        self.phase_idx = phase_idx
        self.phase.start()

    @property
    def phase(self):
        return self.phase_order[self.phase_idx]

    def is_phase(self, phase_name):
        return self.phase.name == phase_name

    def is_phases(self, phase_name_list):
        return any(self.is_phase(phase_name) for phase_name in phase_name_list)

    def get_phase_image(
        self,
        size=(320, 50),
        get_text_func=lambda phase: remove_suffix(phase.name, "Phase"),
        get_color_func=lambda phase: 255,
    ):
        phase_image = np.full(
            size[::-1] + (3,), get_color_func(self.phase), dtype=np.uint8
        )

        cv2.putText(
            phase_image,
            get_text_func(self.phase),
            (5, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        return phase_image
