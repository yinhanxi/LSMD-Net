cur_dir=`dirname "$0"`
tool=$cur_dir/../tools/

#DS=stereo_sceneflow_driving
DS=stereo_livox
#DS=monocular_livox
#DS=monocular_boxdepth
#DS=fusion_kitti

DM_F=$cur_dir/data_module.py:$DS
TP=train
$tool/browse_dataset.py -t $TP $DM_F $*
