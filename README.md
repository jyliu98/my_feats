# my_feats

do_minigpt4.sh提取minigpt4的frame特征

环境和minigpt4一样，在./minigpt4_feat/enviroment.yml里

do_minigpt4.sh中设置frame的位置，数据集，gpu——id等

将./minigpt4_feat/ckpt中的模型换为真实的模型（分别是blip2权重和MLP权重），放置在81的/data3/ljy/minigpt4/ckpt/。

do_blip2_minigpt4.sh提取minigpt4的blip2和minigpt4的frame特征
