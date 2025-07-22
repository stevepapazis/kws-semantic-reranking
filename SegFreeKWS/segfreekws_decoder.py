import argparse
from pathlib import Path

import numpy as np
import torch.cuda
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from utils.gw_dataset import GWDataset
from utils.iam_dataset import IAMDataset


import time

from utils.auxilary_functions import average_precision

from torchvision.ops.boxes import box_iou
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF

from scan_functions import form_kws, generate_maps, intersection_metric, phoc_like




# KWS typically evaluated on a reduced character set without punctuations etc.
# reduced character set to be in line with KWS methods
reduced_charset = True

# character set - to be pruned if reduced character set is true
classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

# use reduced character set to be in line with KWS methods
reduced_charset = True

def reduced(istr):
    return ''.join([c if (c.isalnum() or c=='_' or c==' ') else '*' for c in istr.lower()])

if reduced_charset:
    classes = reduced(classes)
    nclasses = ''
    for c in classes:
        if c in nclasses:
            continue
        else:
            nclasses += c
    classes = nclasses

cdict = {c:i for i,c in enumerate(classes)}
icdict = {i:c for i,c in enumerate(classes)}


def get_dataset(dataset_path, dataset_name, test_fold=None):
    if dataset_name == 'iam':
        dataset_constructor = IAMDataset
    elif dataset_name == 'gw':
        dataset_constructor = lambda *_args, **_kwargs: GWDataset(*_args, **_kwargs, fold=test_fold)
    else:
        raise NotImplementedError
    return dataset_constructor(dataset_path, subset='all', segmentation_level='form', fixed_size=None, transforms=None)
    

class Decoder:
    def __init__(self, model_path, gpu_id=0):
        # load CNN
        self.cnn = torch.load(model_path)
        self.cnn.cuda(gpu_id)
        self.cnn.eval()
        
    def decode(self, page, bboxes, dilate_r = 0.15):
        """bboxes should be given in the xyxy format"""
        if page.dtype != torch.float:
            page = tvF.to_dtype(page, dtype=torch.float, scale=True)
        
        device = next(self.cnn.parameters()).device
        page = (1-page).squeeze().to(device)
        
        transcriptions = []
        
        for bbox in bboxes:
            y,x,y2,x2 = bbox
            w = y2 - y
            h = x2 - x
            y1 = max(int(y - dilate_r * w),0)
            y2 = int(y + (1+dilate_r)*w)
            x1 = max(int(x - 5), 0)
            x2 = int(x + h + 5)
            word_img = page[x1:x2, y1:y2].unsqueeze(0).unsqueeze(0)
                
            with torch.no_grad():
                ycnt, yctc = self.cnn(word_img)
                predicted_charseq = yctc.argmax(2).squeeze() 
                predicted_charseq = torch.atleast_1d(predicted_charseq)
            
            chars = []
            try:
                for j, v in enumerate(predicted_charseq):
                    if j == 0 or v != predicted_charseq[j - 1]:
                        chars.append(icdict[v.item()])
            except TypeError as e:
                import traceback
                print(traceback.format_exc())
                print("typeerror: iteration over a 0-d tensor",e)
                print("segfree_decoder.py, line 96")
                print(j)
                print(v)
                print(bbox)
                print(predicted_charseq)
            transcription = "".join(chars).replace("_", "").strip()
                
                
                
            transcriptions.append(transcription)
            
            #debug
            #import matplotlib.pyplot as plt
            #print(f"{transcription=}, {bbox=}")
            #plt.imsave("tmp.jpg", 1-word_img.cpu().squeeze(),cmap="gray")
            #input("press key")
            
        return transcriptions


if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    # - train arguments
    parser.add_argument('--bounding-box-files', required=True, nargs='+', help="Files with dictionaries that map page names to its bounding boxes")
    parser.add_argument('--output-files', required=True, nargs='+')
    parser.add_argument('--gpu_id', '-gpu', action='store', type=int, default='0',
                        help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
    parser.add_argument('--dataset_path', action='store', type=str, default='../../datasets/')
    parser.add_argument('--model_path',  action='store', type=str, default='./saved_models/temp_best.pt')
    parser.add_argument('--dataset',  action='store', type=str, default='iam')
    parser.add_argument('--test_fold', action='store', type=int)

    args = parser.parse_args()

    if len(args.bounding_box_files) != len(args.output_files):
        parser.error("the number of input and output files should be the same")
    bbox_files = args.bounding_box_files
    output_files = args.output_files 

    form_test_set = get_dataset(args.dataset_path, args.dataset, args.test_fold)

    decoder = Decoder(args.model_path,  args.gpu_id)
 
    get_page_index = {page_name: n for n,page_name in enumerate(form_test_set.page_names)}


    for bbox_file, output_file in zip(bbox_files, output_files):
        bboxes_dict = np.load(bbox_file, allow_pickle=True).item()
        
        for page_name in bboxes_dict.keys():
            page_index = get_page_index[page_name]
            page_img = 1-form_test_set[page_index][0]
            for entry in bboxes_dict[page_name]:
                bboxes = entry["bboxes"]
                entry["transcriptions"] = decoder.decode(page_img, bboxes)
            
        np.save(output_file, bboxes_dict)
