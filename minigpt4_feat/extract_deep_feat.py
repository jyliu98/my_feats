import os
import sys
import json
import time
import torch
import logging
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse

from constant import *
from utils.generic_utils import Progbar
from data_provider import ImageDataset
from torchvision import transforms

from PIL import Image
import numpy as np
import torch

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *
import pdb


logger = logging.getLogger(__file__)
logging.basicConfig(
    format="[%(asctime)s - %(filename)s:line %(lineno)s] %(message)s",
    datefmt='%d %b %H:%M:%S')
logger.setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    cudnn.benchmark=True
    torch.backends.cudnn.benchmark = True


def get_feature_name(model_name, oversample):
    feat = model_name
    return '%s,os' % (feat) if oversample else '%s' % (feat)
    # if 'clip' in model_name:
    #     return '%s,os' % (feat) if oversample else '%s' % (feat)
    # else:
    #     return '%s,%s,os' % (feat,layer) if oversample else '%s,%s' % (feat, layer)

def process(options, collection):
    rootpath = options.rootpath
    model_dir = os.path.join(rootpath, options.model_dir)
    oversample = options.oversample
    model_name = options.model_name

    # layer = 'avgpool'
    batch_size = options.batch_size
    # feat_name = get_feature_name(model_name, layer, oversample)
    feat_name = get_feature_name(model_name, oversample)

    feat_dir = os.path.join(rootpath, collection, 'FeatureData', feat_name)
    id_file = os.path.join(feat_dir, 'id.txt')
    feat_file = os.path.join(feat_dir, 'id.feature.txt')
    id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt')
    
    if options.split != "-1":
        id_path_file = os.path.join(rootpath, collection, 'id.imagepath.txt.split', options.split)
        feat_file += options.split
        print('id_path_file:%s \nfeature_file:%s'%(id_path_file, feat_file))

    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)


    for x in [id_file, feat_file]:
        if os.path.exists(x):
            if not options.overwrite:
                if options.resume:
                    logger.info('%s exists. resume', x)
                else:
                    logger.info('%s exists. skip', x)
                    return 0
            else:
                logger.info('%s exists. overwrite', x)



    cfg = Config(options)
    model_config = cfg.model_cfg
    model_config.device_8bit = options.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    model.eval()
    model = model.to(device)
    dataset = ImageDataset(id_path_file, oversample=oversample)

    dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)
    logger.info('%d images', len(dataset))

    if options.resume:
        feat_file += 'resume'
    print("feat_file: %s"%feat_file)

    fw = open(feat_file, 'w')

    progbar = Progbar(len(dataset))
    start_time = time.time()
    for image_ids, image_tensor in dataloder:
        
        batch_size = len(image_ids)
        if torch.cuda.is_available():
            image_tensor = image_tensor.to(device)

        with torch.no_grad():
            output2, output1 = model.encode_img(image_tensor)

        # if oversample:
        #     output = output.view(batch_size, ncrops, -1).mean(1)
        # else:
        #     output = output.view(batch_size, -1)
        # output1:blip2, output2:minigpt4
        output = torch.mean(output2, dim=1, keepdim=False)
        target_feature = output.cpu().numpy()
        for i, image_id in enumerate(image_ids):
            fw.write('%s %s\n' % (image_id, ' '.join( ['%g' % (x) for x in target_feature[i] ])))

        progbar.add(batch_size)
    elapsed_time = time.time() - start_time
    logger.info('total running time %s', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    fw.close()

    
        #  >>> input, target = batch # input is a 5d tensor, target is 2d
        #  >>> bs, ncrops, c, h, w = input.size()
        #  >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
        #  >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops



def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    from optparse import OptionParser

    parser = OptionParser(usage="""usage: %prog [options] collection""")
    parser.add_option("--rootpath", default=ROOT_PATH, type="string", help="rootpath (default: %s)" % ROOT_PATH)
    parser.add_option("--overwrite", default=0, type="int", help="overwrite existing file (default=0)")
    parser.add_option("--gpu", default=0, type="int", help="gpu id (default: 0)")
    parser.add_option("--oversample", default=0, type="int", help="oversample (default: 0)")
    parser.add_option("--model_dir",default=DEFAULT_MODEL_DIR, type="string")
    parser.add_option("--model_name",default=DEFAULT_MODEL_NAME, type="string")
    parser.add_option("--batch_size",default=1024, type="int")
    parser.add_option("--split",default='', type="string", help="deal one split part of entire collection")
    parser.add_option("--resume", default=0, type="int", help="oversample (default: 0)")
    parser.add_option("--cfg-path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_option("--gpu-id", type=int, default=7, help="specify the gpu to load the model.")
    parser.add_option(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    (options, args) = parser.parse_args(argv)
    if len(args) < 1:
        parser.print_help()
        return 1
    
    print(json.dumps(vars(options), indent = 2))
    
    return process(options, args[0])


if __name__ == '__main__':
    sys.exit(main())

