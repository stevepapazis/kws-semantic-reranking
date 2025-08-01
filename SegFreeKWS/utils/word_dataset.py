import io,os
import numpy as np
from skimage import io as img_io
import torch
from torch.utils.data import Dataset
from os.path import isfile
from skimage.transform import resize
from utils.auxilary_functions import image_resize, centered
import tqdm

#import sys
#import numpy as np
#from PIL import Image

import cv2

class WordLineDataset(Dataset):
    #
    # TODO list:
    #
    #   Create method that will print data statistics (min/max pixel value, num of channels, etc.)
    '''
    This class is a generic Dataset class meant to be used for word- and line- image datasets.
    It should not be used directly, but inherited by a dataset-specific class.
    '''
    def __init__(self,
        basefolder: str = 'datasets/',                #Root folder
        subset: str = 'all',                          #Name of dataset subset to be loaded. (e.g. 'all', 'train', 'test', 'fold1', etc.)
        segmentation_level: str = 'line',             #Type of data to load ('line' or 'word')
        fixed_size: tuple =(128, None),               #Resize inputs to this size
        transforms: list = None,                      #List of augmentation transform functions to be applied on each input
        character_classes: list = None,               #If 'None', these will be autocomputed. Otherwise, a list of characters is expected.
        ):
        self.basefolder = basefolder
        self.subset = subset
        self.segmentation_level = segmentation_level
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.setname = None                             # E.g. 'IAM'. This should coincide with the folder name
        self.stopwords = []
        self.stopwords_path = None
        self.character_classes = character_classes

    def __finalize__(self):
        '''
        Will call code after descendant class has specified 'key' variables
        and ran dataset-specific code
        '''
        assert(self.setname is not None)
        if self.stopwords_path is not None:
            for line in open(self.stopwords_path):
                self.stopwords.append(line.strip().split(','))
            self.stopwords = self.stopwords[0]
        os.makedirs(self.basefolder + '/saved_datasets', exist_ok=True)
        save_file = self.basefolder + '/saved_datasets/{}_{}_{}.pt'.format(self.subset, self.segmentation_level, self.setname) #dataset_path + '/' + set + '_' + level + '_IAM.pt'

        if isfile(save_file) is False:
            print("Generating the dataset from main_loader...")
            data = self.main_loader(self.subset, self.segmentation_level)
            torch.save(data, save_file)   #Uncomment this in 'release' version
        else:
            print("\n\nUSING THE CACHED VERSION!!\n\n")
            print(f"Loading the dataset found in {save_file}")
            data = torch.load(save_file)
        self.data = data
        print("#images : " + str(len(data)))
        if self.character_classes is None:
            res = set()
             #compute character classes given input transcriptions
            for _,transcr, _ in tqdm.tqdm(data):
                res.update(list(transcr))
            res = sorted(list(res))
            print('Character classes: {} ({} different characters)'.format(res, len(res)))
            self.character_classes = res
        #END FINALIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index][0]
        transcr = " " + self.data[index][1] + " "

        bbox = self.data[index][2]
        if self.segmentation_level == 'fword':

            #'''
            if self.subset == 'train':
                si = int(max(0, bbox[0] - np.random.uniform(-.05, .05) * bbox[2]))
                ei = int(min(img.shape[0], bbox[0] + bbox[2] + np.random.uniform(-.05, .05) * bbox[2]))
                sj = int(max(0, bbox[1] - np.random.uniform(-.1, .01) * bbox[3]))
                ej = int(min(img.shape[1], bbox[1] + bbox[3] + np.random.uniform(-.1, .01) * bbox[3]))
            else:
                si, ei = bbox[0], bbox[0] + bbox[2]
                sj, ej = bbox[1], bbox[1] + bbox[3]

            bbox_mask = np.zeros(img.shape[:])
            bbox_mask[si:ei, sj:ej] = 1

            timg = np.zeros([img.shape[0], img.shape[1], 3])
            timg[:, :, 0] = img
            timg[:, :, 1] = bbox_mask

            img = timg

        if self.fixed_size is not None:
            fheight, fwidth = self.fixed_size[0], self.fixed_size[1]

            if self.subset == 'train':
                # random resize at training !!!
                nwidth = int(np.random.uniform(.5, 2.0) * img.shape[1])
                nheight = int((np.random.uniform(.8, 1.2) * img.shape[0] / img.shape[1]) * nwidth)
            else:
                nheight, nwidth = img.shape[0], img.shape[1]

            # small pads!!
            hm, wm = int(.1 * fheight), int(.1 * fwidth)
            nheight, nwidth = max(hm, min(fheight-hm, nheight)), max(wm, min(fwidth-wm, nwidth))
            img = image_resize(img, height=int(1.0 * nheight), width=int(1.0 * nwidth))


            img = centered(img, (fheight, fwidth), border_value=0.0)

            if self.transforms is not None:
                for tr in self.transforms:
                    if np.random.rand() < .9:
                        img = tr(img)

            if self.segmentation_level == 'fword':
                bbox = img[...,1]
                img = img[..., 0]

            # add distractor lines!

            if np.random.uniform() > .75:
                sp = (0, np.random.randint(img.shape[0]))
                ep = (img.shape[1] - 1, np.random.randint(img.shape[0]))

                img = cv2.line(np.asarray(255 * img, dtype=np.uint8), sp, ep,
                               min(255, np.random.randint(int(1.2 * 255 * img.max()))),
                               np.random.randint(1, 5)) / 255.0

        else:
            fwidth = max(64, img.shape[1])
            fheight = max(64, img.shape[0])
            img = centered(img, (fheight, fwidth), border_value=0.0)

        img = torch.Tensor(img).float().unsqueeze(0)

        if 'word' in self.segmentation_level:
            if bbox is not None:
                bbox = torch.Tensor(bbox).float().unsqueeze(0)
            else:
                bbox = torch.Tensor(np.ones((img.shape[0], img.shape[1]))).float().unsqueeze(0)

        return img, transcr, bbox

    def main_loader(self, subset, segmentation_level) -> list:
        # This function should be implemented by an inheriting class.
        raise NotImplementedError

    def check_size(self, img, min_image_width_height, fixed_image_size=None):
        '''
        checks if the image accords to the minimum and maximum size requirements
        or fixed image size and resizes if not

        :param img: the image to be checked
        :param min_image_width_height: the minimum image size
        :param fixed_image_size:
        '''
        if fixed_image_size is not None:
            if len(fixed_image_size) != 2:
                raise ValueError('The requested fixed image size is invalid!')
            new_img = resize(image=img, output_shape=fixed_image_size[::-1], mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        elif np.amin(img.shape[:2]) < min_image_width_height:
            if np.amin(img.shape[:2]) == 0:
                print('OUCH')
                return None
            scale = float(min_image_width_height + 1) / float(np.amin(img.shape[:2]))
            new_shape = (int(scale * img.shape[0]), int(scale * img.shape[1]))
            new_img = resize(image=img, output_shape=new_shape, mode='constant')
            new_img = new_img.astype(np.float32)
            return new_img
        else:
            return img

    def print_random_sample(self, image, transcription, id, as_saved_files=True):
        import random    #   Create method that will show example images using graphics-in-console (e.g. TerminalImageViewer)
        from PIL import Image
        # Run this with a very low probability
        x = random.randint(0, 10000)
        if(x > 5):
            return
        def show_image(img):
            def get_ansi_color_code(r, g, b):
                if r == g and g == b:
                    if r < 8:
                        return 16
                    if r > 248:
                        return 231
                    return round(((r - 8) / 247) * 24) + 232
                return 16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)
            def get_color(r, g, b):
                return "\x1b[48;5;{}m \x1b[0m".format(int(get_ansi_color_code(r,g,b)))
            h = 12
            w = int((img.width / img.height) * h)
            img = img.resize((w,h), Image.ANTIALIAS)
            img_arr = np.asarray(img)
            h,w  = img_arr.shape #,c
            for x in range(h):
                for y in range(w):
                    pix = img_arr[x][y]
                    print(get_color(pix, pix, pix), sep='', end='')
                    #print(get_color(pix[0], pix[1], pix[2]), sep='', end='')
                print()
        if(as_saved_files):
            Image.fromarray(np.uint8(image*255.)).save('/tmp/a{}_{}.png'.format(id, transcription))
        else:
            print('Id = {}, Transcription = "{}"'.format(id, transcription))
            show_image(Image.fromarray(255.0*image))
            print()

class LineListIO(object):
    '''
    Helper class for reading/writing text files into lists.
    The elements of the list are the lines in the text file.
    '''
    @staticmethod
    def read_list(filepath, encoding='ascii'):
        if not os.path.exists(filepath):
            raise ValueError('File for reading list does NOT exist: ' + filepath)

        linelist = []
        if encoding == 'ascii':
            transform = lambda line: line.encode()
        else:
            transform = lambda line: line

        with io.open(filepath, encoding=encoding) as stream:
            for line in stream:
                line = transform(line.strip())
                if line != '':
                    linelist.append(line)
        return linelist

    @staticmethod
    def write_list(file_path, line_list, encoding='ascii',
                   append=False, verbose=False):
        '''
        Writes a list into the given file object

        file_path: the file path that will be written to
        line_list: the list of strings that will be written
        '''
        mode = 'w'
        if append:
            mode = 'a'

        with io.open(file_path, mode, encoding=encoding) as f:
            if verbose:
                line_list = tqdm.tqdm(line_list)

            for l in line_list:
                #f.write(unicode(l) + '\n')   Python 2
                f.write(l + '\n')

