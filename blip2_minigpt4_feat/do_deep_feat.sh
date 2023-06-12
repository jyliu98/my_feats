gpu_id=$1 
rootpath=$2
overwrite=$3
raw_feat_name1=$4
raw_feat_name2=$5
test_collection=$6
model_dir=$7
model_name1=$8
model_name2=$9
batch_size=${10}
split=${11}
echo $split

if [ "$oversample" -eq 1 ]; then
    raw_feat_name1=${raw_feat_name1},raw_feat_name2=${raw_feat_name2},os
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
                        --model_name1 $model_name1 \
                        --model_name2 $model_name2 \
                        --batch_size ${batch_size} \
                        --split $split \
                        --resume 0  \
                        --gpu-id ${gpu_id} \
                        # 继续中断的提取




feat_dir1=$rootpath/${test_collection}/FeatureData/$raw_feat_name1
feat_file1=$feat_dir1/id.feature.txt
feat_dir2=$rootpath/${test_collection}/FeatureData/$raw_feat_name2
feat_file2=$feat_dir2/id.feature.txt

# exit
if [ -f ${feat_file1} ]; then
    python ${BASEDIR}/txt2bin.py 0 $feat_file1 0 $feat_dir1 --overwrite $overwrite
    # rm $feat_file
fi
if [ -f ${feat_file2} ]; then
    python ${BASEDIR}/txt2bin.py 0 $feat_file2 0 $feat_dir2 --overwrite $overwrite
    # rm $feat_file
fi