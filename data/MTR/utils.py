import os
import sys
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# print(BASE_DIR)
sys.path.append('/home/tjtanaa/Documents/Github/AB3DMOT')

from data.MTR.config import config

def id2str(id):
    return config['class_map'][id]

def str2id(classname):
    str2id_map = {v: k for k, v in config['class_map'].items()}
    return str2id_map[classname]

def sort_list(directory_list, charbefore=20):

    def func(x):
        return x[:charbefore]+x[charbefore:][:-4].zfill(4)
    return sorted(directory_list,key=func)


def load_directory_list_from_path(path, suffix='_dir'):
    """
        @return:
            A list containing absolute path to directories
    """
    directory_list = []
    if(os.path.exists(path)):
        directory_list = [os.path.join(path, directory) for directory in os.listdir(path)
                                if (os.path.isdir(os.path.join(path, directory)) and suffix in directory) ]
    else:
        raise FileNotFoundError
    
    return directory_list

def load_filenames_from_path(path, extension='.bin'):
    """
        helper function to aggregate the filenames into a single list containing <absolute path to the file>

    """
    sorted_filenames_list = []
    if(os.path.exists(path)):
        directory_list = load_directory_list_from_path(path)
        
        for directory in directory_list:
            filename_list = [filename for filename in os.listdir(os.path.join(path, directory))
                                        if (os.path.isfile(
                                                os.path.join(path, 
                                                os.path.join(directory, filename)
                                                )) and extension in filename)  ]
            
            filename_list = sort_list(filename_list)

            sorted_filenames_list += [os.path.join(path, os.path.join(directory, filename)) for filename in filename_list]
    else:
        raise FileNotFoundError
    
    return sorted_filenames_list

def load_data_filenames_from_path(path):
    """
        aggregate the filenames into a single list containing <absolute path to the file>

    """
    
    return load_filenames_from_path(path, extension='.bin')

def load_annotation_filenames_from_path(path):
    """
        aggregate the filenames into a single list containing <absolute path to the file>
    """
    return load_filenames_from_path(path, extension='.json')

def load_filenames_from_directory(directory_path, extension='.bin'):
    """
        @params:
        directory_path: absolute path to the directory
    """
    sorted_filenames_list = []
    if(os.path.exists(directory_path)):
        filename_list = [filename for filename in os.listdir(directory_path)
                                    if (os.path.isfile(
                                            os.path.join(directory_path, filename)
                                            ) and extension in filename)  ]
        
        filename_list = sort_list(filename_list)

        sorted_filenames_list += [os.path.join(directory_path, filename) for filename in filename_list]
    else:
        raise FileNotFoundError
    
    return sorted_filenames_list


def load_annotations_from_file_in_mtr_format(filepath):
    """
        filepath is the absolute path to the annotations

        MTR data format
        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    with open(filepath, 'r') as f:
        json_obj = json.load(f)
        # print(json_obj)
        bounding_boxes = json_obj['bounding_boxes']
        
        # filter out noisy annotations
        # and convert the data to kitti MOTS data format
        
        # []
        annotation_list = []
        track_id = -1
        for bboxes in bounding_boxes:
            if bboxes['center']['z'] is None or bboxes['height'] is None or bboxes['height'] < 0.001 \
                or bboxes['width'] < 0.001 or bboxes['length'] < 0.001:
                continue
            # annotation = [frame_id, -1]
            annotation = []
            # print("type: ", str2id(bboxes['object_id']))
            # object_type = bboxes['object_id'] # suppress as 'pedestrian'
            object_type = 'pedestrian'
            # truncated = -1
            # occluded = -1
            # alpha = -1
            # bbox2d = [-1, -1, -1, -1]
            dimensions = [bboxes['height'], bboxes['width'], bboxes['length']]
            location = [bboxes['center']['x'], bboxes['center']['y'], bboxes['center']['z']]
            rotation_y = bboxes['angle']

            annotation.append(object_type)
            # annotation.append(truncated)
            # annotation.append(occluded)
            # annotation.append(alpha)
            # annotation += bbox2d
            annotation += dimensions
            annotation += location
            annotation.append(rotation_y)
            annotation_list.append(annotation)
        return annotation_list

def load_annotations_from_file_in_kittimot_format(filepath, frame_id):
    """
        filepath is the absolute path to the annotations

        kitti MOTS data format
        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    frame        Frame within the sequence where the object appearers
        1    track id     Unique tracking id of this object within this sequence
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                            truncated refers to the object leaving image boundaries.
                    Truncation 2 indicates an ignored object (in particular
                    in the beginning or end of a track) introduced by manual
                    labeling.
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1    score        Only for results: Float, indicating confidence in
                            detection, needed for p/r curves, higher is better.
    """
    with open(filepath, 'r') as f:
        json_obj = json.load(f)
        # print(json_obj)
        bounding_boxes = json_obj['bounding_boxes']
        
        # filter out noisy annotations
        # and convert the data to kitti MOTS data format
        
        # []
        annotation_list = []
        track_id = -1
        for bboxes in bounding_boxes:
            if bboxes['center']['z'] is None or bboxes['height'] is None or bboxes['height'] < 0.001 \
                or bboxes['width'] < 0.001 or bboxes['length'] < 0.001:
                continue
            annotation = [frame_id, -1]
            # print("type: ", str2id(bboxes['object_id']))
            # object_type = bboxes['object_id'] # suppress as 'pedestrian'
            object_type = 'pedestrian'
            truncated = -1
            occluded = -1
            alpha = -1
            bbox2d = [-1, -1, -1, -1]
            dimensions = [bboxes['height'], bboxes['width'], bboxes['length']]
            location = [bboxes['center']['x'], bboxes['center']['y'], bboxes['center']['z']]
            rotation_y = bboxes['angle']

            annotation.append(object_type)
            annotation.append(truncated)
            annotation.append(occluded)
            annotation.append(alpha)
            annotation += bbox2d
            annotation += dimensions
            annotation += location
            annotation.append(rotation_y)
            annotation_list.append(annotation)
        return annotation_list


def convert_mtr_to_kittimot_format(data_list, frame_id):
    """
        filepath is the absolute path to the annotations

        kitti MOTS data format
        #Values    Name      Description
        ----------------------------------------------------------------------------
        1    frame        Frame within the sequence where the object appearers
        1    track id     Unique tracking id of this object within this sequence
        1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                            'Misc' or 'DontCare'
        1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                            truncated refers to the object leaving image boundaries.
                    Truncation 2 indicates an ignored object (in particular
                    in the beginning or end of a track) introduced by manual
                    labeling.
        1    occluded     Integer (0,1,2,3) indicating occlusion state:
                            0 = fully visible, 1 = partly occluded
                            2 = largely occluded, 3 = unknown
        1    alpha        Observation angle of object, ranging [-pi..pi]
        4    bbox         2D bounding box of object in the image (0-based index):
                            contains left, top, right, bottom pixel coordinates
        3    dimensions   3D object dimensions: height, width, length (in meters)
        3    location     3D object location x,y,z in camera coordinates (in meters)
        1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
        1    score        Only for results: Float, indicating confidence in
                            detection, needed for p/r curves, higher is better.
    """
    annotation_list = []
    track_id = -1
    for data in data_list:
        annotation = [frame_id, -1]
        # print("type: ", str2id(bboxes['object_id']))
        object_type = data[0]
        truncated = -1
        occluded = -1
        alpha = -1
        bbox2d = [-1, -1, -1, -1]
        dimensions = data[1:4]
        location = data[4:7]
        rotation_y = data[7]

        annotation.append(object_type)
        annotation.append(truncated)
        annotation.append(occluded)
        annotation.append(alpha)
        annotation += bbox2d
        annotation += dimensions
        annotation += location
        annotation.append(rotation_y)
        annotation_list.append(annotation)
    return annotation_list

def convert_kittimot_to_ab3dmot_format(data_list):
    """
        convert KITTI MOTS format to AB3DMOT format

        AB3DMOT format
        =============================================================================================
        Frame	Type	2D BBOX (x1, y1, x2, y2)	Score	3D BBOX (h, w, l, x, y, z, rot_y)	Alpha
        0	2 (car)	726.4, 173.69, 917.5, 315.1	13.85	1.56, 1.58, 3.48, 2.57, 1.57, 9.72, -1.56	-1.82


        @params:
        data_list: a list containing data in KITTI MOTs format
    """

    ab3dmot_data_list = []

    for data in data_list:
        annotation = []
        annotation.append(data[0])
        annotation.append(str2id(data[2]))
        annotation.append(80) # max scores as it is human annotated
        annotation += data[6:17]
        annotation.append(data[5])
        ab3dmot_data_list.append(annotation)

    return ab3dmot_data_list


if __name__ == "__main__":
    datasource_path = "/home/tjtanaa/Documents/AKK/Project4-MTR"
    datapath = os.path.join(datasource_path, 'Data')
    labelpath = os.path.join(datasource_path, 'Label')

    print("========================== Data Loading =======================")
    print("data path \t: ", datapath)
    print("annotation path \t: ", labelpath)

    data_filename_list = load_data_filenames_from_path(datapath)
    annotation_filename_list = load_annotation_filenames_from_path(labelpath)
    print("Total number of data filenames \t: ", len(data_filename_list))
    print("Total number of anno filneames \t: ", len(annotation_filename_list))


    print("================= MTR format to Kitti MOTS format =============")
    data_sample_mtr_format = load_annotations_from_file_in_mtr_format(annotation_filename_list[100])
    data_sample_kittimot_format = np.array(convert_mtr_to_kittimot_format(data_sample_mtr_format, 100))

    print("Before converting MTR format to Kitti format:")
    # print(data_sample_mtr_format)
    print("After convertion:")
    print(data_sample_kittimot_format.shape)

    print("================= Kitti MOTS format to AB3DMOT format ==========")
    data_sample_kittimot_format = load_annotations_from_file_in_kittimot_format(annotation_filename_list[100], 100)
    data_sample_ab3dmot_format = np.array(convert_kittimot_to_ab3dmot_format(data_sample_kittimot_format))

    print("Before converting Kitti format to AB3DMOT format:")
    # print(data_sample_kittimot_format)
    print("After convertion:")
    print(data_sample_ab3dmot_format.shape)


