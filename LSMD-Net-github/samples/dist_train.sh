cur_dir=`dirname $0`

tool=$cur_dir/../tools/

NET=rt_stereo
MODEL=$cur_dir/model_${NET}.py:$NET

DS=sceneflow_driving

BATCH_SIZE=8
GPUS=4
GPU_IDS=0,1,2,3

WORK_DIR=$cur_dir/../work_dirs

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=29500 \
    $tool/train.py --gpu_ids=$GPU_IDS --launcher pytorch --no_validate --ds $DS $MODEL --batch_size $BATCH_SIZE -w $WORK_DIR $OPTS $*
