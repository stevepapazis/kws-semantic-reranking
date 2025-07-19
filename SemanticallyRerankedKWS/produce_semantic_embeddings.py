import argparse
from pathlib import Path
import sys
import numpy as np
import polars as pl
from tqdm import tqdm
import skimage.io as skio

from collections import defaultdict

import torch
import torchvision.transforms.v2.functional as tvF

from sentence_transformers import CrossEncoder, SentenceTransformer

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import transformers.trainer_utils

from semantically_reranked_kws import load_ground_truth_GW, load_ground_truth_IAM

import evaluate


device = torch.device("cuda:0")



class TrOCRDecoder:
    def __init__(self, model_name):
        self.classes = "0123456789abcdefghijklmnopqrstuvwxyz "

        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
                
    def reduce(self, word):
        return "".join([c for c in word.lower() if c in self.classes])
                
    def decode(self, page, bboxes, dilate_r=0.):#0.05): #TODO: what about this value?
        page = tvF.grayscale_to_rgb(page)
        if page.dtype != torch.uint8:
            page = tvF.to_dtype(page, dtype=torch.uint8, scale=True)
    
        transcriptions = []
        
        for bbox in bboxes:
            y,x,y2,x2 = bbox
            w = y2 - y
            h = x2 - x
            y1 = max(int(y - dilate_r * w),0)
            y2 = int(y + (1+dilate_r)*w)
            x1 = max(int(x - 5), 0)
            x2 = int(x + h + 5)
            word_img = page[..., x1:x2, y1:y2]
            
            with torch.no_grad():
                pixel_values = self.processor(word_img, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(device)
                generated_ids = self.model.generate(pixel_values)
                transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
                transcription = self.reduce(transcription)#.split()
                
                transcriptions.append(transcription)
            
#            #debug
#            import matplotlib.pyplot as plt
#            print(f"{transcription=}, {bbox=}")
#            plt.imsave("tmp.jpg", (np.ubyte(word_img.squeeze().permute(1,2,0).cpu())),cmap="gray")
#            input("press key")
            
        return transcriptions


def load_gw_page(path_to_gw, page_name):
    page = skio.imread(path_to_gw/f"GW/pages/{page_name}.tif")
    return tvF.to_image(page)
    
def load_iam_page(path_to_iam, page_name):
    page = skio.imread(path_to_iam/f"IAM/forms/{page_name}.png")
    return tvF.to_image(page)

def compute_cer(decoder, ground_truth_df, test_pages_dict):
    cer_metric = evaluate.load("cer")
    
    for (query, page_name), group_df in tqdm(list(ground_truth_df.group_by("query", "page"))):
        gt_bboxes = group_df['gt_bbox'].to_list()
        transcriptions = decoder.decode(test_pages_dict[page_name], gt_bboxes)
        cer_metric.add_batch(predictions=transcriptions, references=[query]*len(gt_bboxes))
    
    return cer_metric.compute() 
    





# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--path-to-gw", type=Path, action="store")
parser.add_argument("--path-to-iam", type=Path, action="store")
parser.add_argument("--path-to-kws-results", type=Path, required=True)
parser.add_argument("--test-fold", type=int)
parser.add_argument("--path-to-segfreekws-module", type=Path, required=True)
parser.add_argument('--path-to-segfreekws-decoder', type=Path, required=True)
parser.add_argument("--path-to-trocr", type=Path, default=None)
parser.add_argument("--keep-top-k", type=int, nargs="?", default=None)
parser.add_argument("--avoid-cache", action="store_true", default=None)
parser.add_argument("--skip-WRN", action="store_true", default=None)
parser.add_argument("--skip-SegFreeKWS", action="store_true", default=None)

args = parser.parse_args()
path_to_gw = args.path_to_gw
path_to_iam = args.path_to_iam
path_to_results = args.path_to_kws_results
test_fold = args.test_fold
path_to_segfreekws_decoder = args.path_to_segfreekws_decoder
segfreekws_location = args.path_to_segfreekws_module
top_k = args.keep_top_k
if args.path_to_trocr is None:
    path_to_trocr = "microsoft/trocr-base-handwritten"
else:
    path_to_trocr = args.path_to_trocr

if path_to_gw is not None and test_fold is not None:
    dataset = f"GW{test_fold}"
elif path_to_iam is not None:
    dataset = "IAM"
else:
    raise ValueError("you have to provivde either the path to GW and a test fold or the path to IAM")

# make SegFreeKWS module location discoverable
sys.path.append(str(segfreekws_location.resolve()))
from segfreekws_decoder import Decoder as SegFreeKWSDecoder



print("Loading the ground truth...")
if dataset!="IAM":
    ground_truth, _ = load_ground_truth_GW(path_to_gw/"GW/ground_truth")
else:    
    ground_truth, _ = load_ground_truth_IAM(path_to_iam/"IAM")
    
if dataset!="IAM":
    print(f"Loading WRN-GW{test_fold} results...")
    wrn_suggestions = pl.read_parquet(path_to_results/f"WRN_{dataset}_results.parquet")

    print(f"Loading SegFreeKWS-GW{test_fold} results...")
    segfreekws_suggestions = pl.read_parquet(path_to_results/f"SegFreeKWS_{dataset}_results.parquet")
else:
    print(f"Loading WRN-IAM results...")
    wrn_suggestions = pl.read_parquet(path_to_results/f"WRN_IAM_results.parquet")

    print(f"Loading SegFreeKWS-IAM results...")
    segfreekws_suggestions = pl.read_parquet(path_to_results/f"SegFreeKWS_IAM_results.parquet")
    

if dataset!="IAM":
    test_page_names = wrn_suggestions["page"].drop_nulls().unique().to_list()
    test_pages = {
        name: load_gw_page(path_to_gw, name) 
            for name in test_page_names if name is not None
    }
else:
    test_page_names = segfreekws_suggestions['page'].drop_nulls().unique().to_list()
    test_pages = {
        name: load_iam_page(path_to_iam, name) 
            for name in test_page_names if name is not None
    }
    
smaller_test_pages = {
    name: tvF.resize(page, (page.shape[-2]//2, page.shape[-1]//2)) 
        for name,page in test_pages.items()
}


ground_truth = ground_truth.filter(pl.col("page").is_in(test_page_names))


print(f"Loading the TrOCR decoder: {path_to_trocr}...")
trocr_decoder = TrOCRDecoder(path_to_trocr)
#cer = 100*compute_cer(trocr_decoder, ground_truth, test_pages)
cer = '"computation was skipped"'
print(f"TrOCR-{dataset} CER = {cer}%")

print(f"Loading the SegFreeKWS decoder...")
segfreekws_decoder = SegFreeKWSDecoder(path_to_segfreekws_decoder)
#cer = 100*compute_cer(
#    segfreekws_decoder, 
#    ground_truth.with_columns( (pl.col("gt_bbox")//2) ), 
#    smaller_test_pages
#)
cer = '"computation was skipped"'
print(f"SegFreeKWS-decoder CER = {cer}%")






def transcribe_WRN_suggestions(suggestion_df):
    suggestion_df = suggestion_df.with_row_index()
    
    indices = []
    transcriptions_trocr = []
    transcriptions_segfreekws = []
    
    for (page_name,), grouped_df in tqdm( 
        suggestion_df.group_by("page"),
        total=len(suggestion_df['page'].unique())
    ):
        if page_name is None: 
            indices.extend(grouped_df['ranking'].to_list())
            transcriptions_trocr.extend([None]*len(grouped_df['ranking']))
            transcriptions_segfreekws.extend([None]*len(grouped_df['ranking']))
            continue
    
        transcriptions_by_bbox = dict()
        
        bboxes = set([tuple(b) for b in grouped_df['bbox'].sort()])
        bboxes = torch.Tensor(list(bboxes))
                
        trocr_per_page_transcriptions = trocr_decoder.decode(test_pages[page_name], bboxes)
        trocr_transcriptions_by_bbox = {
            tuple([int(i) for i in b]): tr 
                for b,tr in zip(bboxes, trocr_per_page_transcriptions)
        }
        
        segfreekws_per_page_transcriptions = segfreekws_decoder.decode(smaller_test_pages[page_name], bboxes//2)
        segfreekws_transcriptions_by_bbox = {
            tuple([int(i) for i in b]): tr 
                for b,tr in zip(bboxes, segfreekws_per_page_transcriptions)
        }
        
        for index, bbox in tqdm(
            grouped_df["index", "bbox"].iter_rows(),
            desc=f"Transcribing WRN-{page_name}",
            total=len(grouped_df),
            leave=False
        ):  
            indices.append(index)
            bbox = tuple(bbox)
            transcriptions_trocr.append(trocr_transcriptions_by_bbox[bbox])
            transcriptions_segfreekws.append(segfreekws_transcriptions_by_bbox[bbox])
    
    transcriptions_df = pl.DataFrame({
        "index": indices,
        "transcription_trocr": transcriptions_trocr,
        "transcription_segfreekws": transcriptions_segfreekws
    })
    
    transcriptions_df.write_parquet(path_to_results/"tmp_WRN_transcriptions.parquet")
    
    joined_df = suggestion_df.join(transcriptions_df, on = ["index"]).drop('index')
            
    return joined_df
    

def transcribe_SegFreeKWS_suggestions(suggestion_df):
    trocr_decoded_transcriptions = []
    segfreekws_decoded_transcriptions = []
        
    for page_name, bbox in tqdm(
        suggestion_df["page", "bbox"].iter_rows(),
        desc=f"Transcribing SegFreeKWS",
        total=len(suggestion_df)
    ):
        if page_name is None:
            trocr_decoded_transcriptions.append(None)
            segfreekws_decoded_transcriptions.append(None)
            continue
            
        bboxes = torch.Tensor([bbox])
        
        trocr_decoded_transcriptions.extend(
            trocr_decoder.decode(test_pages[page_name], bboxes, dilate_r=0.2) 
        )
        segfreekws_decoded_transcriptions.extend(
            segfreekws_decoder.decode(smaller_test_pages[page_name], bboxes//2, dilate_r=0.2)
        )
    
    np.save("tmp_transcriptions_trocr.npy", trocr_decoded_transcriptions)
    np.save("tmp_transcriptions_segfreekws.npy", segfreekws_decoded_transcriptions)
    
    return suggestion_df.with_columns(
        transcription_trocr = pl.Series(trocr_decoded_transcriptions),
        transcription_segfreekws = pl.Series(segfreekws_decoded_transcriptions)
    )


if (
    args.avoid_cache 
    and args.skip_WRN is None
    and 
    ("transcription_trocr" not in wrn_suggestions or "transcription_segfreekws" not in wrn_suggestions)
):
    print("Extracting transcriptions for the WRN suggestions")
    wrn_suggestions = transcribe_WRN_suggestions(wrn_suggestions)
    wrn_suggestions.write_parquet(path_to_results/f"WRN_{dataset}_results.parquet")

if (
    args.avoid_cache 
    and args.skip_SegFreeKWS is None
    and
    ("transcription_trocr" not in segfreekws_suggestions or "transcription_segfreekws" not in segfreekws_suggestions)
):
    print("Extracting transcriptions for the SegFreeKWS suggestions")
    segfreekws_suggestions = transcribe_SegFreeKWS_suggestions(segfreekws_suggestions)
    segfreekws_suggestions.write_parquet(path_to_results/f"SegFreeKWS_{dataset}_results.parquet")    



def compute_semantic_similarity_on_WRN(suggestion_df, semantic_encoder, similarity_key):
    grouped_df = suggestion_df.group_by("query", "page").all()

    queries = grouped_df["query"].to_list()
    query_embeddings = semantic_encoder.encode(queries)
    transcription_keys = ["transcription_trocr", "transcription_segfreekws"]    
    similarity_scores = []
        
    for transcription_key in transcription_keys:
        transcription_embeddings = dict()
        for page in grouped_df["page"].unique():
            if page is None: continue
            
            transcriptions = np.atleast_1d(
                np.unique(
                    grouped_df.filter(page=page)[transcription_key]
                              .explode()
                              .unique()
                )
            )
            encoded_page_transcriptions = semantic_encoder.encode(transcriptions)
            transcription_embeddings[page] = {
                tr:v for tr,v in zip(transcriptions, encoded_page_transcriptions)
            }
        
        scores = []
        for q_emb, (page, transc_list) in zip(query_embeddings, grouped_df["page", transcription_key].iter_rows()):

            if page is None:
                scores.append(None)
                continue
            
            scores.append(
                semantic_encoder.similarity(
                    q_emb, 
                    [transcription_embeddings[page][tr] 
                        for tr in transc_list]
                ).squeeze(0).tolist()
            )
        similarity_scores.append(scores)
                
    return grouped_df.with_columns(**{
        similarity_key + "_trocr_decoded": pl.Series(similarity_scores[0]),
        similarity_key + "_segfreekws_decoded": pl.Series(similarity_scores[1])
    }).explode(pl.all().exclude("query", "page"))
    
    
def compute_semantic_similarity_on_SegFreeKWS(suggestion_df, semantic_encoder, similarity_key):
    queries = suggestion_df["query"].unique().to_list()
    query_embeddings = dict(zip(queries, semantic_encoder.encode(queries)))
    
    similarity_scores = [[], []]
    score_cache = dict()
    
    transcription_keys = ["transcription_trocr", "transcription_segfreekws"]    
    
    for i, transcription_key in enumerate(transcription_keys):
        transcriptions = suggestion_df[transcription_key].drop_nulls().unique().to_list()
        transcription_embeddings = dict(zip(transcriptions, semantic_encoder.encode(transcriptions)))
    
        for query, transcription in suggestion_df["query", transcription_key].iter_rows():
            if transcription is None:
                similarity_scores[i].append(None)
                continue
            
            similarity_scores[i].append(
                score_cache.setdefault(
                    (query, transcription),
                    semantic_encoder.similarity(
                        query_embeddings[query],
                        transcription_embeddings[transcription]
                    )
                )
            )
    
    return suggestion_df.with_columns(**{
        similarity_key + "_trocr_decoded": pl.Series(similarity_scores[0]),
        similarity_key + "_segfreekws_decoded": pl.Series(similarity_scores[1])
    })




semantic_encoders = [
    "all-mpnet-base-v2",
    "all-MiniLM-L12-v2",
    "stsb-roberta-base", 
]


for n, model_name in enumerate(semantic_encoders):
    print(f"Loading the semantic encoder {model_name}...")
    semantic_encoder = SentenceTransformer(model_name, device=device)
    semantic_encoder.eval()
    
    if args.skip_WRN is None:
        wrn_suggestions = compute_semantic_similarity_on_WRN(
            wrn_suggestions, 
            semantic_encoder, 
            f"semantic_similarity_using_'{model_name}'"
        )
    
    if args.skip_SegFreeKWS is None:
        segfreekws_suggestions = compute_semantic_similarity_on_SegFreeKWS(
            segfreekws_suggestions, 
            semantic_encoder,
            f"semantic_similarity_using_'{model_name}'"
        )
    
if args.skip_WRN is None:
    wrn_suggestions.write_parquet(path_to_results/f"WRN_{dataset}_results.parquet")
if args.skip_SegFreeKWS is None:
    segfreekws_suggestions.write_parquet(path_to_results/f"SegFreeKWS_{dataset}_results.parquet")    

