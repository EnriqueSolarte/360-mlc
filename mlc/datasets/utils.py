
from mlc.datasets.mp3d_fpe import MP3D_FPE

def load_mvl_dataset(cfg):
    """
    Loads a dataset class defined in cfg.dataset.eval
    """

    if cfg.target_dataset == 'mp3d_fpe':
        # ! loading MP3D-FPE dataset class
        dataset = MP3D_FPE(cfg)
    else:
        raise NotImplementedError("")
    
    return dataset

    
    