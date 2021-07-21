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

from flask import Flask, render_template, Response
import cv2
import time

from collections import defaultdict

app = Flask(__name__)
# app = Flask(__name__, static_url_path='/static')


@app.route('/')
def index():
    return render_template('index.html')

def gen_sample_frame():
    
    count = 0

    while True:
        images = np.zeros((512,512,3))

        images[count%512, :, :] = [255,255,255]
        time.sleep(1.0/30)
        count +=1
        # print("count: ", count)
        yield images
        

def gen():

    frame_gen = DemoDataset().read_from_zmq()
    # frame_gen = gen_sample_frame()
    while True:
        frame = next(frame_gen)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


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

        # Socket to talk to server
        print("Connecting to hello world server...")

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
        self.mot_tracker = AB3DMOT()
        self.trajectory_dict=defaultdict(list)
        self.velocity_dict=defaultdict(list)
        self.speed_dict={}
        self.accel_vect_dict={}
        self.accel_scalar_dict={}

        print("Listening zmq data from {}".format(self.tcp))

        # parameters to convert x, y to pixel location u,v
        self.point_cloud_range = np.array([-50.4, -50.4, 0.0, 50.4, 50.4, 8.0])

        self.pc_dimensions = self.point_cloud_range[3:] - self.point_cloud_range[:3]
        self.pc_resolution = np.array([0.1, 0.1, 0.15])
        self.pc_max_steps = np.ceil(self.pc_dimensions / self.pc_resolution).astype(np.int32)

        # print(pc_max_steps)

        self.color_map = {
            0: (255,0,0),
            1: (0,255,0),
            2: (0,0,255),
            3: (255,255,0),
            4: (255,0,255),
            5: (0,255,255),
            6: (255,255,255)
        }  
        self.num_color = len(self.color_map)



    def __len__(self):
        # return len(self.sample_file_list)
        return 10000

    def read_from_zmq(self):
        total_frames = 0
        while True:
            # print("reading")
            message = self.socket.recv()


            r_trackletpacket = TrackletsPacket.TrackletsPacket.GetRootAs(message, 0)
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

            
            dets = np.asarray(bbox_attr) if len(bbox_attr) > 0 else np.zeros((1,7))
            additional_info = np.asarray(score_array)[:,np.newaxis] if len(bbox_attr)>0 else np.zeros((1,1))
            # get irrelevant information associated with an object, not used for associationg
            # ori_array = seq_dets[seq_dets[:, 0] == frame, -1].reshape((-1, 1))		# orientation
            # other_array = seq_dets[seq_dets[:, 0] == frame, 1:7] 		# other information, e.g, 2D box, ...
            # additional_info = np.concatenate((ori_array, other_array), axis=1)		

            # dets = seq_dets[seq_dets[:,0] == frame, 7:14]            # h, w, l, x, y, z, theta in camera coordinate follwing KITTI convention
            dets_all = {'dets': np.asarray(bbox_attr), 'info': additional_info}

            # important
            # start_time = time.time()
            trackers = self.mot_tracker.update(dets_all)
            # cycle_time = time.time() - start_time
            # total_time += cycle_time

            print("frame: ", total_frames, " ndet: ", nTracklets, " ntrackers: ", len(trackers))
            # saving results, loop over each tracklet			
            viz_img = np.zeros( (self.pc_max_steps[1], self.pc_max_steps[0], 3) ,dtype=np.int8)
            for d in trackers:
                bbox3d_tmp = d[0:7]       # h, w, l, x, y, z, theta in camera coordinate
                id_tmp = d[7]
                ori_tmp = d[8]
                # print("id_tmp: ", id_tmp, "ori_temp: ", ori_tmp)

                self.trajectory_dict[id_tmp].append([
                    (bbox3d_tmp[5]- self.point_cloud_range[1]) // self.pc_resolution[1],
                    (bbox3d_tmp[3] - self.point_cloud_range[0]) // self.pc_resolution[0] 
                    ])
                
                self.velocity_dict[id_tmp]
                self.speed_dict

                # Polygon corner points coordinates
                pts = np.array(
                            self.trajectory_dict[id_tmp],
                            np.int32)
                
                pts = pts.reshape((-1, 1, 2))
                
                isClosed = False
                
                # Blue color in BGR
                color = self.color_map[id_tmp % self.num_color]
                
                # Line thickness of 2 px
                thickness = 2
                
                # Using cv2.polylines() method
                # Draw a Blue polygon with 
                # thickness of 1 px
                viz_img = cv2.polylines(viz_img, [pts], 
                                    isClosed, color, thickness)




            total_frames += 1
        
            yield viz_img

if __name__ == '__main__':

    demo_dataset = DemoDataset()
    total_time, total_frames = 0.0, 0
    seq_count = 0
    mot_tracker = AB3DMOT()

    # app.run(host='0.0.0.0', debug=True)
    app.run(host='0.0.0.0', port=5012)
    exit()
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