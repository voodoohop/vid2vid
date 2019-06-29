### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import numpy as np
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
if opt.dataset_mode == 'temporal':
    opt.dataset_mode = 'test'

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
input_nc = 1 if opt.label_nc != 0 else opt.input_nc

save_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
print('Doing %d frames' % len(dataset))

input_frame = False
for data,i in enumerate(dataset):
    if i == 0:
        input_frame = data

print(input_frame)
model.fake_B_prev = None
for i in range(0, opt.how_many):
    if i >= opt.how_many:
        break    
    #if input_frame['change_seq']:


    _, _, height, width = input_frame['A'].size()
    A = Variable(input_frame['A']).view(1, -1, input_nc, height, width)
    B = Variable(input_frame['B']).view(1, -1, opt.output_nc, height, width) if len(input_frame['B'].size()) > 2 else None
    inst = Variable(input_frame['inst']).view(1, -1, 1, height, width) if len(input_frame['inst'].size()) > 2 else None
    generated = model.inference(A, B, inst)
    
    input_frame['A'] = generated[1]
    input_frame['B'] = generated[0]

    if opt.label_nc != 0:
        real_A = util.tensor2label(generated[1], opt.label_nc)
    else:
        c = 3 if opt.input_nc == 3 else 1
        real_A = util.tensor2im(generated[1][:c], normalize=False)    
        
    visual_list = [('real_A', real_A), 
                   ('fake_B', util.tensor2im(generated[0].data[0]))]
    visuals = OrderedDict(visual_list) 
    img_path = "./results/res_{:04d}.png".format(1)
    print('process image... %s' % img_path)
    visualizer.save_images(save_dir, visuals, img_path)