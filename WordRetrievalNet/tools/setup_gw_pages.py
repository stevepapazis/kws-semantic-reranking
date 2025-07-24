#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

import argparse
import pathlib
import shutil
import string
import numpy as np
import cv2
import tqdm

import augmentation




EMBEDDING_UNI_GRAMS = string.ascii_lowercase + string.digits

def word_filter(word):
    word = str(word).lower()
    for remove_char in [c for c in word if c not in EMBEDDING_UNI_GRAMS]:
        word = word.replace(remove_char, "")
    return word


class GWDB:
    def __init__(self, path_to_db):
        self.path            = path_to_db
        self.original_images = self.path/"images"
        self.documents       = self.path/"documents"
        self.test_set        = np.load(self.path/"test_set.npy")        
        self.validation_set  = np.load(self.path/"validation_set.npy")

        self.images = dict()
        self.labels = dict()
        
        for mode in ["train", "validation", "test", "gen"]:
            self.images[mode] = self.path/mode/"images"
            self.labels[mode] = self.path/mode/"labels"

            for folder in ["images", "labels", "vis"]:
                (self.path/mode/folder).mkdir(
                    parents=True,
                    exist_ok=True
                )

    def split_train_test(self):#, train_percentage=None, validation_percentage=None, seed=None):
        doc_names = [doc.stem for doc in sorted(self.documents.iterdir())]
        train_set = np.array(doc_names)[~(self.test_set|self.validation_set)]
        test_set = np.array(doc_names)[self.test_set]

        print("Setting up train, validation and test sets...")
        for name in tqdm.tqdm(doc_names):
            original = self.original_images/f"{name}.tif"
            if name in train_set:
                mode = "train"
            elif name in test_set:
                mode = "test"
            else:
                mode = "validation"
            destination = self.images[mode]/f"{name}.tif"
            shutil.copyfile(original, destination)

    def _get_x1y1x2y2x3y3x4y4(self, coords):
        xmin, ymin, xmax, ymax = coords
        x1, y1 = xmin, ymin
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        x4, y4 = xmin, ymax
        return x1, y1, x2, y2, x3, y3, x4, y4

    def load_coords_and_labels(self, doc_path):
        coordinates = np.loadtxt(doc_path, dtype=np.int64, usecols=(0,1,2,3))
        labels      = np.loadtxt(doc_path, dtype=np.str_, usecols=4)
        return coordinates, labels
    
    def extract_labels(self, mode, is_vis=True):
        path2mode      = self.path/mode
        all_words_file = (path2mode/"all_words.txt").open("w", encoding="utf-8")
        
        print(f"Extracting labels in {mode} set...")
        for img in tqdm.tqdm(sorted(self.images[mode].iterdir())):
            doc_id = img.stem
            doc    = self.documents/f"{doc_id}.gtp"
            label_file = (self.labels[mode]/f"{doc_id}.txt").open("w", encoding="utf-8")
            
            for coords,label in zip(*self.load_coords_and_labels(doc)):
                label = word_filter(label)
                if label == "": continue
                
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
            doc_id = img.stem
            doc    = self.labels[mode]/f"{doc_id}.txt"
            
            coordinates = np.loadtxt(doc, dtype=np.int64, usecols=(0,1,4,5), delimiter=",")
            labels      = np.loadtxt(doc, dtype=np.str_, usecols=8, delimiter=",")
            
            image_name = f"{doc_id}." + ("tif" if mode!="gen" else "jpg")
            image = cv2.imread(self.images[mode]/image_name)
            image = self.place_labels(image, coordinates, labels)
            cv2.imwrite(self.path/mode/f"vis/{image_name}", image)
           
    def augment(self, amount_of_gen_images, workers):
        if amount_of_gen_images is None:
            amount_of_gen_images = 2000
        if workers is None: 
            workers = 4
        augmentation.augment_dataset(self.path, amount_of_gen_images, workers, label_at_col=9)     
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", metavar="path-to-GW-database")
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
#        # required=True
#    )
    parser.add_argument(
        "-a", "--amount", 
        help="how many augmented examples to produce",
        type=int
    )
    
    args = parser.parse_args()
    
    path2GW = pathlib.Path(args.path)
    
    gwdb = GWDB(path2GW)
    gwdb.split_train_test()#args.train_set_percentage)

    for mode in ["train", "validation", "test"]:
        gwdb.extract_labels(mode)
        gwdb.compute_word_frequencies(mode)
        
    gwdb.augment(amount_of_gen_images=args.amount, workers=args.workers)
    
    if args.visualize:
        for mode in ["train", "validation", "test", "gen"]:
            if mode == "train":
                num_to_vis = 10
            elif mode == "validation":
                num_to_vis = 5
            elif mode == "test":
                num_to_vis = 5
            else:
                num_to_vis = 10
            gwdb.visualize(mode, num_to_vis=num_to_vis)
