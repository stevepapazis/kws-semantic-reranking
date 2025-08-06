import argparse
from pathlib import Path

import sys
import os
import numpy as np
import torch
import torch.cuda
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

from PIL import Image

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
import datasets
import evaluate
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers import default_data_collator

from skimage.io import imread, imsave

rng = np.random.default_rng()


class AugmentedGWWordImages(Dataset):
    def __init__(self, path_to_gw, test_fold, split, processor, max_word_images=None, max_target_length=128):
        self.processor = processor
        self.max_target_length = max_target_length
        self.path_to_gw = path_to_gw / f"cv{test_fold}"
        self.path_to_word_images = self.path_to_gw / f"word_images"
        self.path_to_word_images.mkdir(parents=True, exist_ok=True)
        self.setup_word_images_in_disk(max_word_images)
            
        self.ds = datasets.load_dataset(
            "imagefolder",
            data_dir=str(self.path_to_word_images/split),
            split="train"
        )
        
    def _load_labels(self, file_location):
        queries = np.loadtxt(
            file_location,
            usecols=-1,
            delimiter=",",
            dtype=str
        )
        bboxes = np.loadtxt(
            file_location,
            usecols=(1,5,0,4),
            delimiter=",",
            dtype=int
        )

        return queries, bboxes
        
    def setup_word_images_in_disk(self, max_word_images=None):
        output = self.path_to_word_images
        print(f"Setting up: {output}", file=sys.stderr)
        lock = output / f"word_images_have_been_set.lock"

        if lock.exists(): return
        
        for case in ["gen", "test", "validation"]:
            print(f"\t./{case}", file=sys.stderr)
            (output / case).mkdir(parents=True, exist_ok=True)
            metadata_csv = ["file_name,label"]
            
            if max_word_images is None:
                label_files = (self.path_to_gw/case/"labels").iterdir()
            else:
                label_files = [i for i,_ in zip(
                    (self.path_to_gw/f"{case}/labels").iterdir(),
                    range(max_word_images)
            )]
            
            for label_file in label_files:
                queries, bboxes = self._load_labels(label_file)
                image = imread(str(self.path_to_gw/f"{case}/images/{label_file.stem}") + (".jpg" if case =="gen" else ".tif"))
                 
                for query,b in zip(queries, bboxes):
                    if case == "gen": 
                        w,h = b[1]-b[0],b[3]-b[2]
                        w*=(rng.random()*0.4+0.9)
                        h*=(rng.random()*0.4+0.9)
                        im = image[b[0]:int(b[0]+w), b[2]:int(b[2]+h)]
                    else:
                        im = image[b[0]:b[1], b[2]:b[3]]
                    output_loc = output/ f"{case}/{query}"
                    output_loc.mkdir(exist_ok=True)
                    img_name = f"{query}-{len(list(output_loc.iterdir()))}.jpg"
                    output_loc = output_loc / img_name
                    imsave(output_loc,im)
                    metadata_csv.append(f"{query}/{img_name},{query}")

            print("writing at", str(output/f"{case}/metadata.csv"), file=sys.stderr)
            (output/f"{case}/metadata.csv").write_text("\n".join(metadata_csv))
        
        np.savetxt(lock, [])
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        image = item["image"].convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        input_ids = self.processor.tokenizer(
            item["label"],
            padding="max_length",
            max_length=self.max_target_length
        ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        input_ids = [id_ if id_ != self.processor.tokenizer.pad_token_id else -100 for id_ in input_ids]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(input_ids)}


class GWWordImages(Dataset):
    def __init__(self, path_to_gw, test_fold, split, processor, max_target_length=128):
        self.path_to_word_images = path_to_gw / f"segmented/word_images{test_fold}"
        self.setup_word_images_in_disk(path_to_gw, test_fold)

        self.ds = datasets.load_dataset(str(self.path_to_word_images), split=split)
        self.processor = processor
        self.max_target_length = max_target_length

    def _load_set(self, file_location):
        queries = np.loadtxt(
            file_location,
            usecols=-1,
            delimiter=" ",
            dtype=str
        )
        page_names = np.loadtxt(
            file_location,
            usecols=(0,),
            delimiter=" ",
            dtype=str
        )
        bboxes = np.loadtxt(
            file_location,
            usecols=(1,2,3,4),
            delimiter=" ",
        )

        return queries, page_names, bboxes

    def setup_word_images_in_disk(self, path_to_gw, test_fold):
        output = self.path_to_word_images
        lock = output / f"../word_images{test_fold}_have_been_set.lock"

        if lock.exists(): 
            print("Found cached dataset", file=sys.stderr)
            return

        set_split = path_to_gw / f"set_split/fold{test_fold+1}"

        for case in ["train", "test"]:
            queries, pages, bboxes = self._load_set(set_split/f"{case}set.txt")

            (output / case).mkdir(parents=True, exist_ok=True)

            metadata_csv = ["file_name,label"]

            for query, img_file, bbox in zip(tqdm(queries, desc=f"{case}{test_fold}"), pages, bboxes):
                im = Image.open(path_to_gw / f"pages/{img_file}.tif")
                bbox = np.hstack([bbox[:2], bbox[:2]+bbox[2:]])
                im = im.crop(bbox)
                output_loc = output / f"{case}/{query}"
                output_loc.mkdir(exist_ok=True)
                img_name = f"{query}-{len(list(output_loc.iterdir()))}.jpg"
                output_loc = output_loc / img_name
                im.save(output_loc)
                metadata_csv.append(f"{query}/{img_name},{query}")

            (output/f"{case}/metadata.csv").write_text("\n".join(metadata_csv))

        np.savetxt(lock, [])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]

        image = item["image"].convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        input_ids = self.processor.tokenizer(
            item["label"],
            padding="max_length",
            max_length=self.max_target_length
        ).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        input_ids = [id_ if id_ != self.processor.tokenizer.pad_token_id else -100 for id_ in input_ids]

        return {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(input_ids)}




def create_cer_metric_with_autodecoder(processor):
    """Creates a cer metric with that decodes the trocr output ids automatically"""
    
    classes = "0123456789abcdefghijklmnopqrstuvwxyz "
    cer_metric = evaluate.load("cer")

    def _add_undecoded_batch(pred_ids, label_ids):
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        pred_str = ["".join([c for c in s.lower() if c in classes]) for s in pred_str]

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        label_str = ["".join([c for c in s.lower() if c in classes]) for s in label_str]

#            print(pred_str, label_str)

        cer_metric.add_batch(predictions=pred_str, references=label_str)

    cer_metric.add_undecoded_batch = _add_undecoded_batch
    
    return cer_metric







if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser()
    # - train arguments
    parser.add_argument('--path-to-gw', action='store', required=True)
    parser.add_argument('--test-fold', action='store', type=int)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--continue-from-checkpoint', action='store_true')
    parser.add_argument('--only-evaluate', action='store_true')

    args = parser.parse_args()
    path_to_gw = Path(args.path_to_gw)
    test_fold = args.test_fold
    output_location = Path(args.output)
    epochs = args.epochs
    continue_training_from_checkpoint = args.continue_from_checkpoint
    only_evaluate = args.only_evaluate


    device = torch.device("cuda:0")


    default_base_model_name = "microsoft/trocr-base-handwritten" #"microsoft/trocr-base-stage1"
    saved_model_location = output_location/f"best_finetuned_gw{test_fold}"

    processor = TrOCRProcessor.from_pretrained(default_base_model_name)
    model = VisionEncoderDecoderModel.from_pretrained(
        saved_model_location if continue_training_from_checkpoint else default_base_model_name
    )
    model.to(device)
        
    print("Started loading datasets...")
    gw_train = AugmentedGWWordImages(path_to_gw, test_fold, "gen", processor, max_word_images=200)
    gw_test = AugmentedGWWordImages(path_to_gw, test_fold, "test", processor)
    gw_val = AugmentedGWWordImages(path_to_gw, test_fold, "validation", processor)
    print("Finished loading datasets")

#    gw_train = GWWordImages(path_to_gw, test_fold, "train", processor)
#    gw_test = GWWordImages(path_to_gw, test_fold, "test", processor)

    train_dataloader = DataLoader(gw_train, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(gw_test, batch_size=4)
    val_dataloader = DataLoader(gw_val, batch_size=4)


    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size

    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 64
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4



    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(.5 * epochs), int(.75 * epochs)], gamma=.1)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #    optimizer,
    #    max_lr=10**-3,
    #    epochs=epochs,
    #    steps_per_epoch=len(train_dataloader)
    #)



    def evaluate_cer_for(model, dataloader):
        cer_metric = create_cer_metric_with_autodecoder(processor)
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                pred_ids = model.generate(batch["pixel_values"].to(device))
                cer_metric.add_undecoded_batch(pred_ids, batch["labels"])
        return cer_metric.compute()




    best_cer = evaluate_cer_for(model, test_dataloader)
    print(f"CER@0: {best_cer*100}%")
    if only_evaluate:
        raise SystemExit
        
    #for epoch in range(epochs):
    #   model.train()

    #   train_loss = 0.0

    #   for batch in tqdm(train_dataloader, desc="Training"):
    #       for k,v in batch.items():
    #           batch[k] = v.to(device)

    #       outputs = model(**batch)
    #       loss = outputs.loss
    #       loss.backward()
    #       optimizer.step()
    #       optimizer.zero_grad()
    ##       scheduler.step()

    #       train_loss += loss.item()

    ##       print(f"{loss.item()/len(batch)=}, {scheduler.get_last_lr()[0]}")

    #   print(f"Loss at epoch {epoch}:", train_loss/len(train_dataloader))

    #   current_cer = evaluate_cer(model, test_dataloader)

    #   print(f"CER@{epoch}: {current_cer*100}%")

    #   if current_cer < best_cer:
    #       model.save_pretrained(saved_model_location)
    #       best_cer = current_cer

    #print(f"Best CER: {best_cer*100}%")

    #exit()


    #TODO what happens with the Seq2SeqTrainer?



    training_args = Seq2SeqTrainingArguments(
        output_dir=output_location/f"Seq2SeqTrainer_finetuned_gw{test_fold}",
        overwrite_output_dir=True,
        num_train_epochs = epochs,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_on_start=True,
        per_device_train_batch_size=4,#8,
        per_device_eval_batch_size=4,#8,
        fp16=True,
        logging_steps=2,
        save_total_limit=1,
        load_best_model_at_end=True,
        save_steps=1000,
        eval_steps=100,
    )

    
    cer_metric = create_cer_metric_with_autodecoder(processor)
    def compute_metrics(pred):
        cer_metric.add_undecoded_batch(pred.predictions, pred.label_ids)
        return {"cer": 100*cer_metric.compute()}


    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor.feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=gw_train,
        eval_dataset=gw_val,
        data_collator=default_data_collator,
    )
    trainer.train()
    
    new_cer = evaluate_cer_for(trainer.model, test_dataloader)
    if new_cer < best_cer:
        trainer.model.save_pretrained(output_location/f"trocr_finetuned_gw_cv{test_fold}")

    print(f"Character error rate on test set: {100*new_cer}%")
    #0.038336078808735505
