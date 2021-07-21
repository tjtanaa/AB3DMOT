# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

from __future__ import print_function
import matplotlib; matplotlib.use('Agg')
import os, numpy as np, time, sys
from AB3DMOT_libs.model import AB3DMOT
# from xinshuo_io import load_list_from_folder, fileparts, mkdir_if_missing

import zmq

import flatbuffers
import schema.tracklet_msg.Bbox as TrackletBbox
import schema.tracklet_msg.Class as TrackletClass
import schema.tracklet_msg.Tracklet as Tracklet
import schema.tracklet_msg.TrackletsPacket as TrackletsPacket


class DemoDataset():
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None, ext='.bin', tcp="tcp://localhost:5558"):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        self.root_path = root_path # save path
        # self.ext = ext
        # data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        # data_file_list.sort()

        # self.data_path = os.path.join(self.root_path, 'Data')

        # self.sample_file_list = get_all_filenames(self.data_path)

        
        # self.context = zmq.Context()

        # Socket to talk to server
        print("Connecting to hello world server...")
        # self.socket = self.context.socket(zmq.PULL)
        # self.socket.connect("tcp://localhost:5560")
        # self.frame_count = 0
        # self.current_dir = str(frame_count)
        # self.current_dir = self.current_dir.zfill(23) + '_dir'

        self.tcp = tcp
        self.context = zmq.Context()

        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 10)
        self.socket.connect(self.tcp)
        self.skip_frame_count = 0

        # a set of variables to analyse the increment in delay over time.
        self.start_datetime=None
        self.delta_array = []
        self.delta_t_array = []

        print("Listening zmq data from {}".format(self.tcp))


        # save_viz_path = '/home/akk/Documents/Github/OpenPCDet/visualization/inference_sample_2'

        # if os.path.exists(save_viz_path):
        #     shutil.rmtree(save_viz_path)
        # os.makedirs(save_viz_path)

        # self.Converter = PointvizConverter(home=save_viz_path)

    def __len__(self):
        # return len(self.sample_file_list)
        return 10000

    def read_from_zmq(self):
        while True:
            # print("reading")
            message = self.socket.recv()


            r_trackletpacket = TrackletsPacket.TrackletsPacket.GetRootAs(message, 0)
            # print(dir(r_trackletpacket))

            # print("num_tracklet: ", r_trackletpacket.TrackletsLength())
            
            # nTracklets = r_trackletpacket.TrackletsLength()

            # for i in range(nTracklets):
            #     r_tracklet = r_trackletpacket.Tracklets(i)
            #     print(r_tracklet.TrackId(), " : ", r_tracklet.Score())

            yield r_trackletpacket
            # # print("read ", message)
            # frame_id = deepcopy(np.frombuffer(message[:4], dtype=np.int32))[0]
            # point_count = deepcopy(np.frombuffer(message[4:8], dtype=np.int32))[0]
            # lidar_timestamp = deepcopy(np.frombuffer(message[8:16], dtype=np.float64))[0]
            # unix_timestamp = deepcopy(np.frombuffer(message[16:24], dtype=np.float64))[0]
            # # compare datetime: flush out message that is out-dated
            
            # current_datetime = datetime.now()
            # current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            # lidar_datetime = datetime.fromtimestamp(lidar_timestamp/1000)
            # lidar_datetime_str = lidar_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            # if self.start_datetime is None:
            #     self.start_datetime = current_datetime

            # delta_t = current_datetime - lidar_datetime
            # # print("dir(delta_t): ", dir(delta_t))
            # # print(delta_t.seconds)
            # # print(delta_t.microseconds)
            # # print(delta_t.seconds*1000 + delta_t.microseconds//1000)
            # self.delta_t_array.append(str(delta_t.seconds*1000 + delta_t.microseconds//1000))
            # # print("delta_t: ", delta_t)
            # # eps = 500
            # # if not ( (delta_t > -timedelta(milliseconds=eps)) &\
            # #     (delta_t  < timedelta(milliseconds=eps)) ):
            # #     self.skip_frame_count +=1
            # #     print("skip: ", self.skip_frame_count)
            # #     # print("lidar_datetime: ", lidar_datetime_str, 
            # #     #     "\t current_datetime: ", current_datetime_str,
            # #     #     "\t skip frame: ", self.skip_frame_count,
            # #     #     "\t eps (ms): ", eps, "\t delta_t: ", delta_t)
            # #     continue

            # # if frame_id % 2 == 0:
            # #     continue

            # info_tag = [frame_id, point_count, lidar_timestamp, unix_timestamp]
            # point_cloud = deepcopy(np.frombuffer(message[24:], dtype=np.float32))

            # data = np.reshape(point_cloud, newshape=(-1, 9))[:, [0,1,2,4]]

            # # self.Converter.compile(task_name="Sample_"+ str(frame_id),
            # #                   coors=data[:,[1,2,0]],
            # #                   intensity=data[:,3])

            # # print(data.shape)
            # # point_cloud[:, :3] = np.matmul(tr_m, np.transpose(point_cloud[:, :3])).transpose()
            # # point_cloud[:, 2] += 5.7

            # # print("data.shape: ", data.shape)
            # data[:, :3] = np.matmul(tr_m, np.transpose(data[:, :3])).transpose()
            # data[:, 2] += 5.7

            # data = trim(data)
            # # print("Generator: ", datetime.fromtimestamp(unix_timestamp))


            # # yield info_tag, coors, features, num_list

            # input_dict = {
            #     'points': data,
            #     'frame_id': frame_id,
                
            # }

            # data_dict = self.prepare_data(data_dict=input_dict)
            # # print("data_dict: ", data_dict)
            # # data_dict['filename'] = bin_file
            # yield info_tag, data_dict


if __name__ == '__main__':



    demo_dataset = DemoDataset()
    total_time, total_frames = 0.0, 0
    seq_count = 0
    mot_tracker = AB3DMOT()
    while True:
    # for _ in range(len(demo_dataset)):

        r_trackletpacket = \
            next(demo_dataset.read_from_zmq())
        
        # print("outside: ", r_trackletpacket)
        nTracklets = r_trackletpacket.TrackletsLength()

        bbox_attr = []
        score_array = []
        for i in range(nTracklets):
            r_tracklet = r_trackletpacket.Tracklets(i)
            # print(r_tracklet.TrackId(), " : ", r_tracklet.Score())
            # bbox = [
            #     r_tracklet.Bbox().H(),
            #     r_tracklet.Bbox().W(),
            #     r_tracklet.Bbox().L(),
            #     r_tracklet.Bbox().X(),
            #     r_tracklet.Bbox().Y(),
            #     r_tracklet.Bbox().Z(),
            #     r_tracklet.Bbox().RotY()
            # ]

            bbox = [
                r_tracklet.Bbox().H(),
                r_tracklet.Bbox().L(),
                r_tracklet.Bbox().W(),
                r_tracklet.Bbox().X(),
                r_tracklet.Bbox().Z(),
                r_tracklet.Bbox().Y(),
                -r_tracklet.Bbox().RotY()
            ]
            bbox_attr.append(bbox)
            score_array.append(r_tracklet.Score())
        # break

        dets = np.asarray(bbox_attr)
        # get irrelevant information associated with an object, not used for associationg
        # ori_array = seq_dets[seq_dets[:, 0] == frame, -1].reshape((-1, 1))		# orientation
        # other_array = seq_dets[seq_dets[:, 0] == frame, 1:7] 		# other information, e.g, 2D box, ...
        # additional_info = np.concatenate((ori_array, other_array), axis=1)		

        # dets = seq_dets[seq_dets[:,0] == frame, 7:14]            # h, w, l, x, y, z, theta in camera coordinate follwing KITTI convention
        dets_all = {'dets': np.asarray(bbox_attr), 'info': np.asarray(score_array)[:,np.newaxis]}

        # important
        start_time = time.time()
        trackers = mot_tracker.update(dets_all)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        print("frame: ", total_frames, " ndet: ", nTracklets, " ntrackers: ", len(trackers))
        # saving results, loop over each tracklet			
        for d in trackers:
            bbox3d_tmp = d[0:7]       # h, w, l, x, y, z, theta in camera coordinate
            id_tmp = d[7]
            ori_tmp = d[8]
            print("id_tmp: ", id_tmp, "ori_temp: ", ori_tmp)
            # type_tmp = det_id2str[d[9]]
            # bbox2d_tmp_trk = d[10:14]
            # conf_tmp = d[14]

            # # save in detection format with track ID, can be used for dection evaluation and tracking visualization
            # str_to_srite = '%s -1 -1 %f %f %f %f %f %f %f %f %f %f %f %f %f %d\n' % (type_tmp, ori_tmp,
            # 	bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
            # 	bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], conf_tmp, id_tmp)
            # save_trk_file.write(str_to_srite)

            # # save in tracking format, for 3D MOT evaluation
            # str_to_srite = '%d %d %s 0 0 %f %f %f %f %f %f %f %f %f %f %f %f %f\n' % (frame, id_tmp, 
            # 	type_tmp, ori_tmp, bbox2d_tmp_trk[0], bbox2d_tmp_trk[1], bbox2d_tmp_trk[2], bbox2d_tmp_trk[3], 
            # 	bbox3d_tmp[0], bbox3d_tmp[1], bbox3d_tmp[2], bbox3d_tmp[3], bbox3d_tmp[4], bbox3d_tmp[5], bbox3d_tmp[6], 
            # 	conf_tmp)
            # eval_file.write(str_to_srite)

        total_frames += 1
        # save_trk_file.close()
    seq_count += 1
    # eval_file.close()    
    print('Total Tracking took: %.3f for %d frames or %.1f FPS' % (total_time, total_frames, total_frames / total_time))