"""Cycle through green phases in network order at a fixed duration."""
import argparse


def green_time_seconds(value: str) -> int:
    duration = int(value)
    if not 3 < duration < 3600:
        raise argparse.ArgumentTypeError("green time must be between 4 and 3599 seconds")
    return duration


class FixedTimeController:
    def __init__(self, phase_num: int, green_time: int = 20):
        if phase_num <= 0:
            raise ValueError("phase_num must be positive")
        self.green_time = green_time_seconds(str(green_time))
        self.phase_num = phase_num
        self.phase_id = 0

    def reset(self) -> None:
        self.phase_id = 0

    def act(self) -> int:
        phase_id = self.phase_id
        self.phase_id = (phase_id + 1) % self.phase_num
        return phase_id
