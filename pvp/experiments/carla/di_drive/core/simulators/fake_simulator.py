'''
Copyright 2021 OpenDILab. All Rights Reserved:
Description:
'''

from .base_simulator import BaseSimulator


class FakeSimulator(BaseSimulator):
    # To do: complete fake simulator
    def __init__(self, cfg):
        super(FakeSimulator, self).__init__(cfg)

    def get_observations(self):
        pass

    def apply_control(self, control):
        pass

    def run_step(self):
        pass
