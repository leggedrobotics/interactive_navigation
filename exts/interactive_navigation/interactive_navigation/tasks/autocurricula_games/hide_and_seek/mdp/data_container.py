import torch


class DataContainer:
    """Data container for the MDP.
    shared between individual modules of the MDP.

    access with env.cfg.data_container"""

    def __init__(self) -> None:
        self.num_obstacles: torch.Tensor = torch.tensor([0.0])
        self.num_obstacles_range: tuple[int, int] = (1, 10)
