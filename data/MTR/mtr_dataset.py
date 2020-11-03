import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# print(BASE_DIR)
sys.path.append('/home/tjtanaa/Documents/Github/AB3DMOT')

from data.MTR.config import config
from data.MTR.utils import load_directory_list_from_path, load_filenames_from_directory
from data.MTR.utils import load_annotations_from_file_in_mtr_format, convert_mtr_to_kittimot_format, convert_kittimot_to_ab3dmot_format


class MTR_Dataset(object):

    def __init__(self, root_dir):
        self._root_dir = root_dir
        self._data_path = os.path.join(root_dir, 'Data')
        self._annotation_path = os.path.join(root_dir, 'Label')
        self._directory_list = load_directory_list_from_path(self._annotation_path)
        self._directory_index = 0
        self._frame_index = 0

    def get_sequences_list(self):
        return self._directory_list

    def get_number_of_sequences(self):
        return len(self._directory_list)

    def get_current_sequence_number(self):
        return self._directory_index
    
    def load_by_directory(self):
        """
            This function will load the data by directory

            @return:
                The generator returns 
                A list of data with MTR format [object_type h w l x y z ry]
        """
        current_directory = self._directory_list[self._directory_index]
        # print("current directory\t: ", current_directory)
        annotation_filename_list = load_filenames_from_directory(current_directory, extension='.json')
        print(len(annotation_filename_list))
        # print("annotation_filename_list\t: ", annotation_filename_list)
        seq_data = []
        for filename in annotation_filename_list:
            sample = load_annotations_from_file_in_mtr_format(filename)
            sample = convert_mtr_to_kittimot_format(sample, self._frame_index)
            sample = convert_kittimot_to_ab3dmot_format(sample)
            # print(len(sample))
            seq_data += sample
            self._frame_index += 1
            if self._frame_index == 561 + 1:
                print(filename)
                print(len(sample))
        
        self._directory_index += 1

        
        return seq_data
    