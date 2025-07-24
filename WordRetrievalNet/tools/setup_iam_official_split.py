#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import argparse
import pathlib
import shutil
import string
import numpy as np
import shapely
import cv2
from multiprocessing import Pool, shared_memory
from itertools import repeat
import tqdm
from xml.dom import minidom



import augmentation




EMBEDDING_UNI_GRAMS = string.ascii_lowercase + string.digits

def word_filter(word):
    word = str(word).lower()
    for remove_char in [c for c in word if c not in EMBEDDING_UNI_GRAMS]:
        word = word.replace(remove_char, "")
    return word


def copy_file(original_and_destination):
    original, destination = original_and_destination
    return shutil.copyfile(original, destination)


#TODO some segmentations are wrong, they're noted as such in the xml files ("err")
#perhaps remove those bounding boxes
class IAMDB_official_split_test:
    def __init__(self, path_to_db, workers):
        self.path            = path_to_db/"official_split"
        self.original_images = path_to_db/"forms"
        self.documents       = path_to_db/"xml"
        
        self.workers = workers or 4
        
        self.images = dict()
        self.labels = dict()
        
        for mode in ["test"]:#["train", "test", "gen"]:
            self.images[mode] = self.path/mode/"images"
            self.labels[mode] = self.path/mode/"labels"
            
            for folder in ["images", "labels", "vis"]:
                (self.path/mode/folder).mkdir(
                    parents=True, 
                    exist_ok=True
                )
                
    def generate_test_set(self):
        doc_names = [doc.stem for doc in self.documents.iterdir()]
        
        test_set = set()
        with open(self.path/"testset.txt") as testset:
            while (line:=testset.readline().strip()):
                test_set.add("-".join(line.split("-")[:2]))
                
        originals_and_destinations = [
            (
                self.original_images/f"{name}.png", 
                self.images[f"test"]/f"{name}.png"
            ) 
            for name in doc_names if name in test_set
        ]
        
#        print("Setting up train and test sets...")
        print("Setting up test set...")
        with Pool(self.workers) as pool:
            pool_iter = pool.imap_unordered(
                copy_file, 
                tqdm.tqdm(originals_and_destinations)
            )
            for _ in pool_iter: pass
    
    def _get_x1y1x2y2x3y3x4y4(self, coords):
        xmin, ymin, xmax, ymax = coords
        x1, y1 = xmin, ymin
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        x4, y4 = xmin, ymax
        return x1, y1, x2, y2, x3, y3, x4, y4
    
    def load_coords_and_labels(self, doc_path):
        doc = minidom.parse(doc_path.open())
        labels = []
        coordinates = []
        
        for line_elem in doc.getElementsByTagName("line"):
            if line_elem.getAttribute("segmentation") == "err": continue
            
            for word_elem in line_elem.getElementsByTagName("word"):
            # for word_elem in doc.getElementsByTagName("word"):
                label = word_filter(word_elem.getAttribute("text"))
                
                if label == "": continue
                
                labels.append(label)
                
                cmp = word_elem.getElementsByTagName("cmp")
                coords = []
                for point in cmp:
                    x = int(point.getAttribute("x"))
                    y = int(point.getAttribute("y"))
                    w = int(point.getAttribute("width"))
                    h = int(point.getAttribute("height"))
                    coords.extend([(x,y), (x+w,y+h)])
                
                if len(coords) > 2:
                    mbb = shapely.Polygon(coords).envelope
                    X,Y = mbb.exterior.xy
                    xmin, ymin, xmax, ymax = min(X), min(Y), max(X), max(Y)
                elif len(coords) == 2:
                    xmin, ymin, xmax, ymax = coords[0][0], coords[0][1], coords[1][0], coords[1][1]
                else:
                    continue    # some examples are not annotated 
                
                coordinates.append(np.array([xmin, ymin, xmax, ymax], dtype=np.int64))

        return coordinates, labels
    
    def extract_labels(self, mode, is_vis=True):
        path2mode      = self.path/mode
        all_words_file = (path2mode/"all_words.txt").open("w", encoding="utf-8")
        
        print(f"Extracting labels in {mode} set...")
        for img in tqdm.tqdm(sorted(self.images[mode].iterdir())):
            doc_id = img.stem
            doc    = self.documents/f"{doc_id}.xml"
            label_file = (self.labels[mode]/f"{doc_id}.txt").open("w", encoding="utf-8")
            
            for coords,label in zip(*self.load_coords_and_labels(doc)):
                x1, y1, x2, y2, x3, y3, x4, y4 = self._get_x1y1x2y2x3y3x4y4(coords)
                all_words_file.write(f"{label}\n")
                label_file.write(f"{x1},{y1},{x2},{y2},{x3},{y3},{x4},{y4},{label}\n")
        
        
    def compute_word_frequencies(self, mode):
        path2mode = self.path/mode
        all_words_file = (path2mode/"all_words.txt")
        
        print(f"Computing unique word occurrences in {mode} set...")
        words = np.loadtxt(all_words_file, dtype=np.str_)
        _, inverse, counts = np.unique(words, return_inverse=True, return_counts=True)
        counts = counts[inverse]
        
        qry_words_count_file = (path2mode/"qry_words.txt")
        qry_words_count_file.write_text(
            "\n".join([
                f"{word}: {count}" for word,count in zip(words, counts)
            ])
        )
        
    def place_labels(self, image, coordinates, labels):
        for coords,label in zip(coordinates, labels):
            x1, y1, _, _, x3, y3, _, _ = self._get_x1y1x2y2x3y3x4y4(coords)
            cv2.rectangle(image, (x1, y1), (x3, y3), (60, 20, 220), 2)
            cv2.putText(image, label, (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 69, 255), 1)
        return image
      
    def visualize(self, mode, num_to_vis=10, seed=0):
        img_names = [img for img in self.images[mode].iterdir()]
        rng = np.random.default_rng(seed)
        img_set = rng.choice(img_names, num_to_vis, replace=False)
        
        print(f"Visualizing {mode} set...")
        for img in tqdm.tqdm(img_set):
            doc = self.labels[mode]/f"{img.stem}.txt"
            coordinates = np.loadtxt(doc, dtype=np.int64, delimiter=",", usecols=(0,1,4,5))
            labels      = np.loadtxt(doc, dtype=np.str_, delimiter=",", usecols=8)
            
            image_name = img.name
            image = cv2.imread(self.images[mode]/image_name)
            image = self.place_labels(image, coordinates, labels)
            cv2.imwrite(self.path/mode/f"vis/{image_name}", image)
           
    def augment(self, amount_of_gen_images):
        if amount_of_gen_images is None:
            amount_of_gen_images = 2000
            
        augmentation.augment_dataset(self.path, amount_of_gen_images, self.workers, label_at_col=9)     
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", metavar="path-to-IAM-database")
    parser.add_argument(
        "-v", "--visualize", 
        help="visualize the ground truth", 
        action="store_true"
    )
    parser.add_argument(
        "-w", "--workers", 
        help="how many workers to use in parallel", 
        type=int,
    )
#    parser.add_argument(
#        "-p", "--train-set-percentage", 
#        help="set train set percentage", 
#        type=float, 
#        required=True
#    )
    parser.add_argument(
        "-a", "--amount", 
        help="how many augmented examples to produce",
        type=int
    )
    
    args = parser.parse_args()
    
    path2IAM = pathlib.Path(args.path)
    
    iamdb = IAMDB_official_split_test(path2IAM, workers=args.workers)
    iamdb.generate_test_set()
    
    for mode in ["test"]:#["train", "test"]:
        iamdb.extract_labels(mode)
        iamdb.compute_word_frequencies(mode)
        
    #iamdb.augment(amount_of_gen_images=args.amount)
    
    if args.visualize:
        for mode in ["train", "test", "gen"]:
            iamdb.visualize(mode)
