import pathlib

from .PARAMETERS import *

class SubjectFiles:
    """
    Helper class to easily switch between files types when needed
    """
    def __init__(self, path:pathlib.Path):
        self.path = path
        self.validate_path()

        self.tmp = SAVE_DIR_TMP
        self.plot = SAVE_DIR_PLOT

    def __repr__(self) -> str:
        return self.path.stem 
    
    def validate_path(self):
        if self.path.is_dir():
            raise ValueError("Only .vhdr files are accepted (got directory instead)")
        if self.path.suffix != '.vhdr':
            raise ValueError(f"Only .vhdr files are accepted (got {self.path.suffixes} instead)")

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def parent(self) -> pathlib.Path:
        return self.path.parent

    @property
    def raw(self) -> pathlib.Path:
        return self.parent / (self.stem + '.vhdr')

    @property
    def epoch(self) -> pathlib.Path:
        return self.tmp / (self.stem + '-epo.fif')
    
    @property
    def gat(self) -> pathlib.Path:
        return self.tmp / (self.stem + '-gat.npy')
    
    def png(self, plot_name: str) -> pathlib.Path:
        return self.plot / (self.stem + '-' + plot_name + '.png')
