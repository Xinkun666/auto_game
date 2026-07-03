from dataclasses import dataclass


@dataclass
class Connector:
    name = "hdc"


@dataclass
class ActionMode:
    """
    Specify ui evnet
    """
    down = "down"
    move = "move"
    up = "up"
