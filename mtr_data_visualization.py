
from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import os, numpy as np, time, sys
from AB3DMOT_libs.model import AB3DMOT
from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing
from data.MTR import * 

import open3d as o3d
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python predict_pedestrian_path.py mtr')
        sys.exit(1)

    result_sha = sys.argv[1]
    save_root = './results'
    det_id2str = {1:'pedestrian', 2:'car', 3:'cyclist'}

    # seq_file_list, num_seq = load_list_from_folder(os.path.join('data/KITTI', result_sha))

    # datapath = os.path.join(dataset_config['database_path'], 'Data')
    # labelpath = os.path.join(dataset_config['database_path'], 'Label')

    print("========================== Data Loading =======================")
    print("Database path \t: ", dataset_config['database_path'])
    dataset = MTR_Dataset_Base(dataset_config['database_path'])
    # print("data path \t: ", datapath)
    # print("annotation path \t: ", labelpath)

    # data_filename_list = load_data_filenames_from_path(datapath)
    # annotation_filename_list = load_annotation_filenames_from_path(labelpath)
    # print("Total number of data filenames \t: ", len(data_filename_list))
    # print("Total number of anno filenames \t: ", len(annotation_filename_list))
    annotation_filename_list = dataset.get_sequences_list()
    num_seq = dataset.get_number_of_sequences()
    total_time, total_frames = 0.0, 0
    save_dir = os.path.join(save_root, result_sha); mkdir_if_missing(save_dir)
    eval_dir = os.path.join(save_dir, 'data'); mkdir_if_missing(eval_dir)
    seq_count = 0
    for seq_file in annotation_filename_list:
        # _, seq_name, _ = fileparts(seq_file)
        # print(seq_file)
        seq_name = seq_file.split('/')[-1]
        eval_file = os.path.join(eval_dir, seq_name + '.txt'); eval_file = open(eval_file, 'w')
        save_trk_dir = os.path.join(save_dir, 'trk_withid', seq_name); mkdir_if_missing(save_trk_dir)

        mot_tracker = AB3DMOT() 
        # seq_dets = np.loadtxt(seq_file, delimiter=',')          # load detections, N x 15
        seq_dets, annotation_filename_list, _ = dataset.load_annotations_by_directory()
        # print(seq_file)
        seq_dets = np.array(seq_dets)
        # print(seq_dets)
        
        point_cloud_filename_list, _ = dataset.load_filenames_by_directory()
        point_cloud_np  = dataset._load_single_point_cloud(point_cloud_filename_list[0])
        print(point_cloud_np.shape)


        pcd = o3d.geometry.PointCloud()
        # point_cloud_np=point_cloud_np[~np.all(np.abs(point_cloud_np) < 0.00001 , axis=1)]
        # point_cloud_np = np.transpose(point_cloud_np)
        # print("point_cloud_np.shape: ", point_cloud_np.shape)
        pcd.points = o3d.utility.Vector3dVector(point_cloud_np[:,:3])
        print(point_cloud_np.shape)
        # colors = np.log10(1e-5 + point_cloud_np[:,4])
        # colors = cv2.equalizeHist(point_cloud_np[:,3])
        colors = point_cloud_np[:,3]
        # min_colors = np.percentile(colors, 10)
        # max_colors = np.percentile(colors, 90)
        min_colors = 10
        max_colors = 170
        colors = np.clip(colors, min_colors, max_colors)
        print("max color: ", max_colors)
        print("min color: ", min_colors)
        # print("Color: Min: ", min_colors, "\t Max: ", max_colors)
        colors_map = plt.get_cmap("viridis")((colors - min_colors) / ((max_colors - min_colors) if (max_colors - min_colors) > 0 else 1))
        # colors_map = plt.get_cmap("inferno")((colors) / (max_colors))
        # print("colors_map.shape: ", colors_map.shape)
        # # print("colors_map[0]: ", colors_map[0])
        # print(colors_map)
        # print("pcd has color: ", pcd.has_colors())
        # print("pcd has points: ", pcd.has_points())
        # pcd.colors = o3d.utility.Vector3dVector(colors_map[:,:3].astype(np.float))
        # pcd.colors = o3d.utility.Vector3dVector(colors[:,:3])
        # o3d.visualization.draw_geometries([pcd])

        # print("pcd has color: ", pcd.has_colors())
        # print("pcd has points: ", pcd.has_points())

        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            return False

        def capture_image(vis):
            image = vis.capture_screen_float_buffer()
            plt.imshow(np.asarray(image))
            plt.show()
            return False

        def save_view_point(pcd, filename):
            # vis = o3d.visualization.Visualizer()
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            opt.point_size = 2
            pcd.colors = o3d.utility.Vector3dVector(colors_map[:,:3].astype(np.float))
            vis.add_geometry(pcd)
            vis.run()  # user changes the view and press "q" to terminate
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            o3d.io.write_pinhole_camera_parameters(filename, param)
            vis.destroy_window()



        def load_view_point(pcd, filename):
            # vis = o3d.visualization.Visualizer()
            vis = o3d.visualization.VisualizerWithKeyCallback()
            vis.create_window()
            ctr = vis.get_view_control()
            param = o3d.io.read_pinhole_camera_parameters(filename)
            pcd.colors = o3d.utility.Vector3dVector(colors_map[:,:3].astype(np.float))
            print("pcd has color: ", pcd.has_colors())
            print("pcd has points: ", pcd.has_points())
            # print("pcd.colors ", np.asarray(pcd.colors))
            vis.add_geometry(pcd)
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            opt.point_size = 2
            ctr.convert_from_pinhole_camera_parameters(param)
            vis.run()
            # vis.poll_events()
            # vis.update_renderer()
            # print("opt.dir")
            # print(dir(opt))
            image = vis.capture_screen_float_buffer()
            # print("image.shape: ", np.asarray(image).shape )
            plt.imsave('./hello.png', np.asarray(image))
            # image = vis.capture_screen_float_buffer()
            # print("image.shape: ", np.asarray(image).shape )
            vis.destroy_window()


        save_view_point(pcd, "viewpoint.json")
        # load_view_point(pcd, "viewpoint.json")
        break