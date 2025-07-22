import numpy as np
from skimage import io as img_io
from utils.word_dataset import WordLineDataset
from utils.auxilary_functions import image_resize, centered
import os
import tqdm

import pathlib


class GWDataset(WordLineDataset):
    def __init__(self, basefolder, subset, fold, segmentation_level, fixed_size, transforms=None):
        super().__init__(basefolder, subset, segmentation_level, fixed_size, transforms)
        self.setname = "GW"#setfolder

        self.images_path = f"{self.baseset_path}/pages"#images"

        self.word_file = f"{self.baseset_path}/words.txt"
        self._setup_word_file()

        self.trainset_file = f"{self.baseset_path}/set_split/fold{fold}/trainset.txt"
        self.testset_file = f"{self.baseset_path}/set_split/fold{fold}/testset.txt"
        self.valset_file = f"{self.baseset_path}/set_split/fold{fold}/valset.txt"
        self._setup_set_splits()

        super().__finalize__()

    @property
    def baseset_path(self):
        return f"{self.basefolder}/{self.setname}"

    def _setup_set_splits(self):
        lock = pathlib.Path(self.baseset_path)/"splits_have_been_set.lock"

        #if lock.exists(): return

        pages = sorted([p.stem for p in pathlib.Path(self.images_path).iterdir()])
        #TODO: what about random split?
        rng = np.random.default_rng(seed=0)
        rng.shuffle(pages)
        pages = np.array(pages)

        set_split = f"{self.baseset_path}/set_split"
        os.makedirs(set_split, exist_ok=True)

        wordlines = np.loadtxt(self.word_file, dtype=np.str_)

        for fold_n in range(1,5):
            fold_dir = f"{set_split}/fold{fold_n}"
            os.makedirs(fold_dir, exist_ok=True)

            test_set = pages[5*(fold_n-1):5*fold_n]
            test_set_indices = np.in1d(wordlines[:, 0], test_set)

            val_set = pages[(5*(fold_n-1+1))%len(pages):(5*(fold_n+1))%len(pages)]
            val_set_indices = np.in1d(wordlines[:, 0], val_set)
            
            train_set_indices = ~np.in1d(wordlines[:, 0], np.concatenate([test_set,val_set]))

            np.savetxt(self.trainset_file, wordlines[train_set_indices], fmt="%s")
            np.savetxt(self.testset_file, wordlines[test_set_indices], fmt="%s")
            np.savetxt(self.valset_file, wordlines[val_set_indices], fmt="%s")

        np.savetxt(lock, pages, fmt="%s", delimiter="\n")


    def _read_documents(self):
        documents = f"{self.baseset_path}/ground_truth"#documents"
        for doc in sorted(os.listdir(documents)):
            if ".gtp" not in doc: continue
            docname = doc.split(".")[0]
            docpath = f"{documents}/{doc}"
            content = np.loadtxt(docpath, dtype=np.str_)
            yield docname, content

    def _setup_word_file(self):
        with open(self.word_file, "w") as word_file:
            for docname, content in self._read_documents():
                for line in content:
                    y1, x1, y2, x2 = [int(s) for s in line[:4]]
                    word_file.write(f"{docname} {y1} {x1} {y2-y1} {x2-x1} {line[4]}\n")

    def main_loader(self, subset, segmentation_level) -> list:

        if segmentation_level not in ["word", "fword", "form"]: raise ValueError

        if subset == 'all':
            valid_set = np.vstack([
                np.loadtxt(self.trainset_file, dtype=str),
                np.loadtxt(self.testset_file, dtype=str)
            ])
            valid_set = valid_set[np.argsort(valid_set[:,0])]
        elif subset == 'train':
            valid_set = np.loadtxt(self.trainset_file, dtype=str)
        elif subset == 'val':
            valid_set = np.loadtxt(self.valset_file, dtype=str)
        elif subset == 'test':
            valid_set = np.loadtxt(self.testset_file, dtype=str)
        else:
            raise ValueError
            
        self.pages_names = np.unique(valid_set[:,0])

        words = []
        form_dict = {}

        for wordline in valid_set:#open(self.word_file):

            info = wordline#.strip().split()
            doc = info[0]

            # if doc not in valid_set: continue

            bbox = np.array([int(info[1]), int(info[2]), int(info[3]), int(info[4])])
            if bbox[2] < 8 and bbox[3] < 8: continue

            transcr = ' '.join(info[5:])

            if segmentation_level == "form":
                form_dict.setdefault(doc, []).append((bbox, transcr))

            words.append((doc, transcr, bbox))

        def load_img(doc):
            img_path = f"{self.images_path}/{doc}.tif"
            docimg = img_io.imread(img_path).astype(np.float32)
            return 1 - docimg/255.0

        data = []

        print("Processing document images")
        if "form" == segmentation_level:
            for doc,info in tqdm.tqdm(form_dict.items()):
                img = load_img(doc)
                img = image_resize(img, height=img.shape[0]//2)

                bboxes = [(bb//2,tr) for bb,tr in info]
                transcr = ""

                data.append((img, transcr, bboxes))
        elif "word" in segmentation_level:
            previous_img_path = None
            for (doc, transcr, bbox) in tqdm.tqdm(words):
                img_path = f"{self.images_path}/{doc}.tif"
                if img_path != previous_img_path:
                    img = load_img(doc)
                    img = image_resize(img, height=img.shape[0]//2)
                    previous_img_path = img_path

                xs, ys, h, w = bbox[:]//2

                if segmentation_level=="word":
                    wordimg = img[ys:(ys+w), xs:(xs+h)]
                    bbox = [0, 0, w, h]

                else:
                    # enlarge bbox
                    hpad = int(2.5 * h / len(transcr))

                    nxs = max(0, xs - hpad)
                    nxe = min(img.shape[1], xs + h + hpad)

                    wpad = 16#//2
                    nys = max(0, ys - wpad)
                    nye = min(img.shape[0], ys + w + wpad)

                    wordimg = img[nys:nye, nxs:nxe]
                    bbox = [ys-nys, xs-nxs, w, h]

                data.append((wordimg, transcr, bbox))

        return data
