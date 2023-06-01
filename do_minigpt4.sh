# source activate clip
# bash do_clip.sh msrvtt10k_blipcaz 1 -1
rootpath=/data1/ljy/VisualSearch1
oversample=0
overwrite=0

batch_size=1024
model_dir=-1
model_name=minigpt4

raw_feat_name=${model_name}
test_collection=msrvtt10k
gpu_id=6
split=-1

bash ./minigpt4_feat/do_deep_feat.sh ${gpu_id} \
                    ${rootpath} \
                    ${oversample} \
                    ${overwrite} \
                    ${raw_feat_name} \
                    ${test_collection} \
                    ${model_dir} \
                    ${model_name} \
                    ${batch_size} \
                    ${split}
