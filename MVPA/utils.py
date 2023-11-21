import pathlib, tomllib

def config_prep(config_file: str | pathlib.Path = pathlib.Path("MVPA/PARAMETERS.toml")) -> dict: 
    """
    Config values need validation and a little preprocessing

    Args:
        config_file (str | pathlib.Path): default 'MVPA/PARAMETERS.toml'
            path pointing to the configuration file

    Returns:
        dict: containing parameters used everywhere in the package
    """
    with open(config_file, 'rb') as f:
        config_ = tomllib.load(f)
    
    # change path strings into pathlib.Path objects, check if directories exist
    for key, val in config_['PATHS'].items():
        config_['PATHS'][key] = pathlib.Path(val)
    
    if not config_['PATHS']['SAVE'].exists():
        raise OSError('Save directory does not exist')

    config_['PATHS']['TMP'] = config_['PATHS']['SAVE'] / '.tmp'
    if not config_['PATHS']['TMP'].exists():
        config_['PATHS']['TMP'].mkdir()

    config_['PATHS']['PLOT'] = config_['PATHS']['SAVE'] / 'plot'
    if not config_['PATHS']['PLOT'].exists():
        config_['PATHS']['PLOT'].mkdir()
    
    return config_

class SubjectFiles:
    def __init__(self, data_path: pathlib.Path, save_path: pathlib.Path):
        """
        Helper class to easily switch between file types when needed

        Args:
            data_path (pathlib.Path): 
                path to the .vhdr BrainVision file containing subject data
            save_path (pathlib.Path):
                path to the directory where results should be saved
        """
        self._validate_data_path(data_path)
        self._validate_save_path(save_path)

        self.path = data_path
        self.tmp = save_path / '.tmp'
        self.plot = save_path / 'plot'

    def __repr__(self) -> str:
        return self.path.stem 
    
    def _validate_data_path(self, p):
        if p.is_dir():
            raise ValueError("Only .vhdr files are accepted (got directory instead)")
        if p.suffix != '.vhdr':
            raise ValueError(f"Only .vhdr files are accepted (got {p.suffixes} instead)")
        
    def _validate_save_path(self, p):
        if not p.is_dir():
            raise ValueError("Invalid path to save directory")

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
    
    @property
    def gat_pvals(self) -> pathlib.Path:
        return self.tmp / (self.stem + '-gat_pval.npy')
    
    @property
    def spat_filter(self) -> pathlib.Path:
        return self.tmp / (self.stem + '-spat_filters-ave.fif')
    
    @property
    def spat_pattern(self) -> pathlib.Path:
        return self.tmp / (self.stem + '-spat_patterns-ave.fif')
    
    def png(self, plot_name: str) -> pathlib.Path:
        return self.plot / (self.stem + '-' + plot_name + '.png')
