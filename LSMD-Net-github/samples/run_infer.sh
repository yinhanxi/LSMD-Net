# cb_f=../work_dirs/rt_stereo_stereo_sceneflow_driving
# model_f=model_rt_stereo.py:rt_stereo_192

# cb_f=../work_dirs/stereonet_disp_stereo_sceneflow_driving
# model_f=model_stereonet_disp:stereonet_disp_192

#cb_f=./work_dirs/cfnet_stereo_sceneflow_driving
#cb_f=./work_dirs/cfnet_stereo_drivingstereo
#model_f=./samples/model_cfnet:cfnet_192

# cb_f=../work_dirs/gwcnet_stereo_sceneflow_driving
# model_f=model_gwcnet:gwcnet_192

# cb_f=../work_dirs/hsmnet_stereo_sceneflow_driving
# model_f=model_hsmnet:hsmnet_192

# cb_f=../work_dirs/cascade_stereo_stereo_sceneflow_driving
# model_f=model_cascade_stereo:cascade_stereo_gwcnet_192

# cb_f=../work_dirs/hitnet_stereo_sceneflow_driving
# model_f=model_hitnet:hitnet_192

#cb_f=./work_dirs/raft_stereo_stereo_sceneflow_driving
#cb_f=./work_dirs/raft_stereo_stereo_drivingstereo
#model_f=./samples/model_raft_stereo:raft_stereo_192

#cb_f=./work_dirs/mac_stereo_stereo_sceneflow_driving
#cb_f=./work_dirs/mac_stereo_stereo_drivingstereo
#model_f=./samples/model_mac_stereo:mac_stereo_192

#cb_f=./work_dirs/coex_stereo_sceneflow_driving
#cb_f=./work_dirs/coex_stereo_drivingstereo
#cb_f=./work_dirs/coex_fusion_kitti
#cb_f=./work_dirs/coex_stereo_livox
#model_f=./samples/model_coex:coex_192

#cb_f=./work_dirs/lsfnet_stereo_sceneflow_driving
#cb_f=./work_dirs/lsfnet_stereo_livox
cb_f=./work_dirs/lsfnet_fusion_kitti
model_f=./samples/model_lsfnet:lsfnet_192

#cb_f=./work_dirs/lidarstereonet_fusion_kitti
#cb_f=./work_dirs/lidarstereonet_stereo_livox
#model_f=./samples/model_lidarstereonet:lidarstereonet_192

#cb_f=./work_dirs/msgchn_stereo_sceneflow_driving
#cb_f=./work_dirs/msgchn_stereo_livox
#cb_f=./work_dirs/msgchn_fusion_kitti
#model_f=./samples/model_msgchn:msgchn_192

#cb_f=./work_dirs/msg_chn_fusion_kitti
#cb_f=./work_dirs/msg_chn_stereo_livox
#model_f=./samples/model_msg_chn:msg_chn_192

#cb_f=./work_dirs/guided_stereo_fusion_kitti
#cb_f=./work_dirs/msg_chn_stereo_livox
#model_f=./samples/model_guided_stereo:guided_stereo_192

#cb_f=./work_dirs/monocular_stereo_sceneflow_driving
#cb_f=./work_dirs/monocular_monocular_boxdepth
#model_f=./samples/model_monocular:monocular_192

#data_f=./data/sceneflow/stereo_sceneflow_driving_test.json
#data_f=./data/boxdepth/monocular_boxdepth_train.json
#data_f=./data/sceneflow/stereo_sceneflow_driving_train.json
#data_f=./data/drivingstereo/drivingstereo_2018-07-10-09-54-03_test.json
#data_f=./data/ADAM/stereo_ADAM_train.json
#data_f=./data/livox/stereo_livox_test.json
#data_f=./data/livox/monocular_livox_test.json
#data_f=./data/drivingstereo/stereo_drivingstereo_test.json
data_f=./data/kitti/stereo_kitti_test.json
#data_f=./data/kitti/fusion_kitti_test.json

./tools/infer.py --shuffle $model_f $cb_f $data_f
#./tools/infer.py $model_f $cb_f $data_f