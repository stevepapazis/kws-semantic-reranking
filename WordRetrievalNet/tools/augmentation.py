#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-


import shutil
from multiprocessing import shared_memory, cpu_count, Pool
from itertools import cycle
import tqdm

import numpy as np

import skimage.filters as skifilters
import skimage.transform as skitransform
import skimage.morphology as skimorphology
import skimage.io as skiio
import skimage.util as skiutil





#### Augment word image ####
def aug_crop(img, rng, l_hpad=0, h_hpad=12, l_vpad=0, h_vpad=12):
    t_img = img < skifilters.threshold_otsu(img)
    nz = t_img.nonzero()
    h_pad = rng.integers(low=l_hpad, high=h_hpad, size=2, endpoint=True)
    v_pad = rng.integers(low=l_vpad, high=h_vpad, size=2, endpoint=True)
    b = [max(nz[1].min() - h_pad[0], 0), max(nz[0].min() - v_pad[0], 0),
         min(nz[1].max() + h_pad[1], img.shape[1]), min(nz[0].max() + v_pad[1], img.shape[0])]
    return img[b[1]:b[3], b[0]:b[2]]


def affine(img, rng, l_shear=-5, h_shear=30, l_rotate=0, h_rotate=1, order=1):
    phi = (rng.uniform(l_shear, h_shear) / 180) * np.pi
    theta = (rng.uniform(l_rotate, h_rotate) / 180) * np.pi
    t = skitransform.AffineTransform(shear=phi, rotation=theta, translation=(-25, -50))
    tmp = skitransform.warp(img, t, order=order, mode='edge', output_shape=(img.shape[0] + 100, img.shape[1] + 100))
    return tmp


def morph(img, rng, l_ksize=3, h_ksize=4):
    choice = rng.integers(2)
    operation = [skimorphology.gray.erosion, skimorphology.gray.dilation][choice]
    ksize = rng.integers(1, [l_ksize, h_ksize][choice], endpoint=True)
    kernel = skimorphology.square(ksize)
    return operation(img, kernel)


def augment(word, rng):
    # assert (word.ndim == 2)
    t = np.zeros_like(word)
    s = np.array(word.shape) - 1
    t[0, :], t[:, 0] = word[0, :], word[:, 0]
    t[s[0], :], t[:, s[1]] = word[s[0], :], word[:, s[1]]
    pad = np.median(t[t > 0])

    tmp = np.ones((word.shape[0] + 8, word.shape[1] + 8), dtype=word.dtype) * pad
    tmp[4:-4, 4:-4] = word
    out = tmp
    out = affine(out, rng)
    out = aug_crop(out, rng)
    out = morph(out, rng)
    out = np.round(out).astype(np.ubyte)
    return out


#### Load data set ####
def load_dataset(path2db, label_at_col=9):
    image_path  = path2db/"train/images/"
    label_path  = path2db/"train/labels/"
    image_files = sorted(image_path.iterdir())
    label_files = sorted(label_path.iterdir())

    data = []
    for image,label_file in zip(image_files, label_files):
        bbxs = np.loadtxt(label_file, dtype=np.int64, usecols=(0,1,4,5), delimiter=",")
        words = np.loadtxt(label_file, dtype=np.str_, usecols=label_at_col-1, delimiter=",")

        gt_boxes = [bb for bb in bbxs]
        regions = [{"id": id, "label": w}  for id,w in enumerate(words)]

        data.append({"path": image, "gt_boxes": gt_boxes, "regions": regions})

    return data


def close_crop_box(im, box):
    gray = im[box[1]:box[3], box[0]:box[2]]
    t_img = gray < skifilters.threshold_otsu(gray)
    h_proj, v_proj = t_img.sum(axis=0), t_img.sum(axis=1)
    x1o = box[0] + max(h_proj.nonzero()[0][0] - 1, 0)
    y1o = box[1] + max(v_proj.nonzero()[0][0] - 1, 0)
    x2o = box[2] - max(h_proj.shape[0] - h_proj.nonzero()[0].max() - 1, 0)
    y2o = box[3] - max(v_proj.shape[0] - v_proj.nonzero()[0].max() - 1, 0)
    obox = (x1o, y1o, x2o, y2o)
    return obox


def augment_word_in_parallel(box, page_shm_name, page_shape, rng):
    page_mem = shared_memory.SharedMemory(name=page_shm_name, create=False)
    page = np.ndarray(shape=page_shape, dtype=np.uint8, buffer=page_mem.buf)

    try:
        box = close_crop_box(page, box)
        word_coords = np.s_[box[1]:box[3], box[0]:box[2]]
        word = page[word_coords]
        aug_word = augment(word, rng)
        aug_word = skitransform.resize(aug_word, word.shape, preserve_range=True)
        page[word_coords] = aug_word
    except:
        pass

    page_mem.close()


#### In-place data augmentation ####
def inplace_augment(path2db, data, workers, samples_per_image=5):
    labels_path = path2db/"train/labels"
    gen_images  = path2db/"gen/images"
    gen_labels  = path2db/"gen/labels"

    workers = min(workers, cpu_count())
    rng = np.random.default_rng()
    worker_rngs = cycle(rng.spawn(workers))

    with Pool(workers) as pool:
        print("Performing inplace augmentation of the train set...")
        print(f"For each document, {samples_per_image} samples will be produced")
        for datum in tqdm.tqdm(data):
            image_path = datum["path"]
            image_name = image_path.stem
            im = skiutil.img_as_ubyte(skiio.imread(image_path, as_gray=True))

            page_shmem = shared_memory.SharedMemory(create=True, size=im.nbytes)
            page = np.ndarray(im.shape, dtype=np.uint8, buffer=page_shmem.buf)

            for j in range(samples_per_image):
                np.copyto(page, im)

                boxes = [
                    (box, page_shmem.name, page.shape, wrng)
                    for box,wrng in zip(datum['gt_boxes'], worker_rngs)
                ]

                pool.starmap(augment_word_in_parallel, boxes)

                skiio.imsave(gen_images/f"{image_name}_{j}.jpg", page)

                shutil.copyfile(
                    labels_path/f"{image_name}.txt",
                    gen_labels/f"{image_name}_{j}.txt"
                )

            page_shmem.close()
            page_shmem.unlink()

        pool.close()
        pool.join()

    print(flush=True)
    print("\nIgnore warnings about memory leakage: https://stackoverflow.com/questions/62748654/python-3-8-shared-memory-resource-tracker-producing-unexpected-warnings-at-appli")



def build_vocab(data):
    """ Builds a set that contains the vocab. Filters infrequent tokens. """
    texts = []
    for datum in data:
        for r in datum['regions']:
            texts.append(r['label'])
    vocab, index = np.unique(texts, return_index=True)
    return vocab, index


def create_background(m, shape, rng, fstd=2, bstd=10):
    canvas = np.ones(shape) * m
    noise = rng.standard_normal(shape) * bstd
    noise = skifilters.gaussian(noise, fstd)  # low-pass filter noise
    canvas += noise
    canvas[canvas>255] = 255
    canvas = np.round(canvas).astype(np.uint8)
    return canvas


# Generate a random page, for parallel page generation
def _generate_random_page(packed_args):
    pageid, shape, bg_color, vocab, labels_to_imagewords, path2db, rng = packed_args

    nwords = 256 #batch size?
    interword_space = 3 #interword space?

    gen_images = path2db/"gen/images"
    gen_labels = path2db/"gen/labels"

    canvas = create_background(bg_color, shape, rng)
    x, y = int(shape[1] * 0.08), int(shape[0] * 0.08)  # Upper left corner of box
    maxy = 0
    img = gen_images/f"fullpage_{pageid}.jpg"
    txt = (gen_labels/f"fullpage_{pageid}.txt").open(mode="w", encoding="utf-8")

    for _ in range(nwords):
        label = rng.choice(vocab, size=1)[0]
        images_of_word = labels_to_imagewords[label]
        word = images_of_word[rng.integers(len(images_of_word))]
        # randomly transform word and place on canvas
        tword = augment(word, rng)

        h, w = tword.shape
        if x + w > int(shape[1] * 0.92):  # done with row?
            x, y = int(shape[1] * 0.08), maxy + interword_space
        if y + h > int(shape[0] * 0.92):  # done with page?
            break

        x1, y1, x2, y2 = x, y, x + w, y + h
        canvas[y1:y2, x1:x2] = tword
        b = [x1, y1, x2, y2]
        x = x2 + interword_space
        maxy = max(maxy, y2)
        txt.write(f"{b[0]},{b[1]},{b[2]},{b[1]},{b[2]},{b[3]},{b[0]},{b[3]},{label}\n")

    skiio.imsave(img, canvas)
    txt.close()


#### Full page data augmentation ####
def fullpage_augment(path2db, data, workers, num_images=1000):
    vocab, _ = build_vocab(data)
    labels_to_imagewords = dict()
    shapes, medians = [], []
    for datum in data:
        im = skiutil.img_as_ubyte(skiio.imread(datum["path"], as_gray=True))

        shapes.append(im.shape)
        medians.append(np.median(im))
        for r,b in zip(datum["regions"], datum["gt_boxes"]):
            word, label = im[b[1]:b[3], b[0]:b[2]], r["label"]
            labels_to_imagewords.setdefault(label,[]).append(word)

    workers = min(workers, cpu_count())
    rng = np.random.default_rng()
    worker_rngs = rng.spawn(workers)

    m = int(np.median(medians))
    bg_colors = m + rng.integers(-10, 11, size=num_images)
    bg_colors[bg_colors>255] = 255

    args = [
        (pid, shape, bgc, vocab, labels_to_imagewords, path2db, wrng)
        for pid,shape,bgc,wrng in zip(range(num_images), cycle(shapes), bg_colors, cycle(worker_rngs))
    ]

    print("Generating random fullpage augmentations of the train set...")
    with Pool(workers) as pool:
        pool_iter = pool.imap_unordered(_generate_random_page, args)
        for _ in tqdm.tqdm(pool_iter, total=len(args)): pass



def augment_dataset(path2db, amount_of_gen_images, workers, label_at_col=9):
    data = load_dataset(path2db, label_at_col)
    num_data = len(data)
    samples_per_image = int(np.round(float(amount_of_gen_images / 2) / num_data))
    nps = amount_of_gen_images - samples_per_image * num_data

    inplace_augment(path2db, data, workers, samples_per_image)
    # print("\nIgnore warnings about memory leakage: https://stackoverflow.com/questions/62748654/python-3-8-shared-memory-resource-tracker-producing-unexpected-warnings-at-appli")
    fullpage_augment(path2db, data, workers, nps) #TODO uncoomment this