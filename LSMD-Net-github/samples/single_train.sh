cur_dir=`dirname $0`

tool=$cur_dir/../tools/

SUBNET=

# NET=rt_stereo
# NET=stereonet_disp

#NET=cfnet

# NET=gwcnet
# NET=hsmnet
# NET=cascade_stereo
# SUBNET=_gwcnet

# NET=hitnet

#NET=coex
#NET=raft_stereo
#NET=mac_stereo
NET=lsfnet
#NET=msgchn
#NET=msg_chn
#NET=monocular
#NET=lidarstereonet
#NET=guided_stereo 

SUFFIX=_192
MODEL=$cur_dir/model_${NET}.py:$NET$SUBNET$SUFFIX

#DS=monocular_boxdepth
#DS=stereo_sceneflow_driving
#DS=stereo_livox
#DS=monocular_livox
#DS=stereo_drivingstereo
DS=fusion_kitti

#GPUS=1,2,4,5,6
GPUS=0,1,2
WORK_DIR=$cur_dir/../work_dirs/${NET}${SUBNET}_${DS}
$tool/train.py $MODEL --ds $DS --gpus $GPUS -w $WORK_DIR $OPTS $*
