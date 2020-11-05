import os
import sys
import json
import numpy as np 

from typing import List, Set, Dict, Tuple, Optional, Any
from typing import Callable, Iterator, Union, Optional, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# print(BASE_DIR)
sys.path.append('/home/tjtanaa/Documents/Github/AB3DMOT')

from data.MTR.config import config
from data.MTR.utils import load_directory_list_from_path, load_filenames_from_directory, load_absolute_directory_list_from_path
from data.MTR.utils import load_annotations_from_file_in_mtr_format, convert_mtr_to_kittimot_format, convert_kittimot_to_ab3dmot_format
from data.MTR.point_cloud_utils import rotation_matrix, transform
from data.MTR.mtr_dataset_base import MTR_Dataset_Base

class MTR_Dataset(MTR_Dataset_Base):
    def __init__(self, root_dir):
        super().__init__(root_dir)

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError


class MTR_Dataset_Random(MTR_Dataset):
    def __init__(self, root_dir):
        super().__init__(root_dir)
