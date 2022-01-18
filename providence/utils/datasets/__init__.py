import torch
from .backblaze import BackblazeDataset
from .nasa import NasaDataSet

from providence.utils import utils as providence_utils

class ProvidenceDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs) -> None:
        """
        Standard PyTorch dataloader, with a change of defaulting the collate function to use
        the jagged-support collate function we have internal to the library
        """
        kwargs['collate_fn'] = providence_utils.collate_fn
        super().__init__(*args, **kwargs)
    
    # copy the parent documentation
    __doc__ = "\nParent documentation:\n"+ torch.utils.data.DataLoader.__doc__


__all__ = [
    'BackblazeDataset', 'NasaDataSet', 'ProvidenceDataLoader'
]