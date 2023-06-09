gpu_id=$1 
rootpath=$2
overwrite=$3
raw_feat_name=$4
test_collection=$5
model_dir=$6
model_name=$7
batch_size=$8
split=${9}
echo $split


if [ "$oversample" -eq 1 ]; then
    raw_feat_name=${raw_feat_name},os
fi  


BASEDIR=$(dirname "$0")
echo 'generate imagepath file'
echo $test_collection
python ${BASEDIR}/generate_imagepath.py ${test_collection} --overwrite 0 --rootpath $rootpath
imglistfile=$rootpath/${test_collection}/id.imagepath.txt


#exit
if [ ! -f $imglistfile ]; then
    echo "$imglistfile does not exist"
    exit
fi


CUDA_VISIBLE_DEVICES=$gpu_id python ${BASEDIR}/extract_deep_feat.py ${test_collection} \
                        --gpu ${gpu_id} \
                        --overwrite $overwrite \
                        --rootpath $rootpath \
                        --model_dir $model_dir \
                        --model_name $model_name \
                        --batch_size ${batch_size} \
                        --split $split \
                        --resume 0  \
                        --gpu-id ${gpu_id} \
                        # 继续中断的提取




feat_dir=$rootpath/${test_collection}/FeatureData/$raw_feat_name
feat_file=$feat_dir/id.feature.txt


# exit
if [ -f ${feat_file} ]; then
    python ${BASEDIR}/txt2bin.py 0 $feat_file 0 $feat_dir --overwrite $overwrite
    # rm $feat_file
fi
