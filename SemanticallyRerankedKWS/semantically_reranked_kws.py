from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import skimage.io as skio
import scipy.spatial.distance as scipydist
import torch
#import torchvision.ops as tvops
import polars as pl
from functools import partial

 
from map_metric import compute_map





def load_ground_truth_GW(ground_truth_dir):
    pages = []
    bboxes = []
    queries = []
    stopwords = []

    for gt_file in ground_truth_dir.iterdir():
        if gt_file.suffix != ".gtp": continue
        contents = np.loadtxt(gt_file, delimiter=" ", dtype=str)
        pages.extend([gt_file.stem]*len(contents))
        bboxes.append(np.int32(contents[:,:4]))
        queries.extend(contents[:,4])
    
    queries = np.array(queries)
    pages = np.array(pages)
    unique_queries, query_inverse = np.unique(queries, return_inverse=True)
    bboxes = np.int32(np.vstack(bboxes))
    
    return pl.DataFrame({
        "query": queries, 
        "page": pages, 
        "gt_bbox": bboxes
    }), stopwords
    
    
def load_ground_truth_IAM(iam_path):
    pages = []
    bboxes = []
    queries = []

    stopwords = np.unique(np.loadtxt(iam_path/"iam-stopwords", delimiter=",", dtype=str))
    
    test_pages = set()
    with open(iam_path/"set_split/testset.txt") as testset:
        while (line:=testset.readline().strip()):
            test_pages.add("-".join(line.split("-")[:2]))
    
    with open(iam_path/"ascii/words.txt") as words:
        while (line:=words.readline().strip()):
            if line[0] == "#": continue
            
            parts = line.split()
            
            if parts[1] != "ok": continue
            
            page = "-".join(parts[0].split("-")[:2])
            
            if page not in test_pages: continue
            
            transcription = "".join(parts[8:]).lower() 
            
            if transcription == "" or transcription in stopwords: continue
            
            transcription = "".join([c for c in transcription
                                            if c in "0123456789abcdefghijklmnopqrstuvwxyz "
                            ])
                            
            if transcription == "" or transcription in stopwords: continue
             
            queries.append(transcription)
            pages.append(page)
            bboxes.append(np.int32([int(parts[3]), int(parts[4]), int(parts[3])+int(parts[5]), int(parts[4])+int(parts[6])]))    
        
    queries = np.array(queries)
    pages = np.array(pages)
    unique_queries, query_inverse = np.unique(queries, return_inverse=True)
    bboxes = np.int32(np.vstack(bboxes))

    return pl.DataFrame({
        "query": queries, 
        "page": pages, 
        "gt_bbox": bboxes,
    }), stopwords


def load_suggestion_dict(transcription_file_npy):
    return np.load(transcription_file_npy, allow_pickle=True).item()




def load_WRN_bbox_suggestions_as_dataframe(wrn_result_location, dataset, test_fold=None):
    if dataset == "GW" and test_fold is not None:
        wrn_result_npy = wrn_result_location/f"predict_result_resnet50_GW{test_fold}.npy"
    elif dataset == "IAM":
        wrn_result_npy = wrn_result_location/"predict_result_resnet50_IAM.npy"
    else:
        raise ValueError("invalid dataset or test_fold")
    data_from_result_npy = load_suggestion_dict(wrn_result_npy)
    db = data_from_result_npy["for_cal_map"]
    page_names = [page.split(".")[0] for page in db]
    
    query_list = []
    page_list = []
    bbox_list = []
    score_list = []
    
    queries = np.hstack([db[page]["gt_word"] for page in db])
    query_embeddings = np.vstack([db[page]["gt_embedding"] for page in db])
    
    for query, ind in zip(*np.unique(queries, return_index=True)):
        query_embedding = query_embeddings[np.newaxis, ind]
        
        for page in db:
            page_name = page.split(".")[0]
            
            bboxes = np.int32(db[page]["pred_coord"][:,[0,1,4,5]])
            bbox_embeddings = np.array(db[page]["pred_embedding"])
            
            cos_distances = scipydist.cdist(query_embedding, bbox_embeddings, metric="cosine").squeeze()
            scores = 1-cos_distances
            
            num_of_retrievals = len(bboxes)
    
            query_list.append([query]*num_of_retrievals)
            page_list.append([page_name]*num_of_retrievals)
            bbox_list.append(bboxes)
            score_list.append(scores)
    
    return pl.DataFrame({
        "query": np.hstack(query_list),
        "page": np.hstack(page_list),
        "bbox": np.vstack(bbox_list),
        "syntactic_similarity": np.hstack(score_list)
    })




def load_SegFreeKWS_bbox_suggestions_as_dataframe(segfreekws_result_location, dataset, test_fold=None):
    if dataset == "GW" and test_fold is not None:
        segfreekws_result_npy = segfreekws_result_location/f"predict_result_segfreekws_GW{test_fold+1}.npy"
    elif dataset == 'IAM':
        segfreekws_result_npy = segfreekws_result_location/f"predict_result_segfreekws_IAM.npy"
    else:
        raise ValueError("invalid dataset or test_fold")
    suggestions = load_suggestion_dict(segfreekws_result_npy)
    test_pages = np.unique([page for query_dict in suggestions.values() for page in query_dict])
    
    query_list = []
    page_list = []
    bbox_list = []
    score_list = []
    
    queries_without_predictions = []
    
    for query in suggestions:
        if len(suggestions[query]) == 0:
            queries_without_predictions.append(query)
            
        for page in suggestions[query]:
            entry = suggestions[query][page]
            bboxes = np.int32(entry["bboxes"]) * 2 # normalise scale, segfreekws halves the images
            scores = entry["scores"]
            num_of_retrievals = len(bboxes)

            query_list.append([query]*num_of_retrievals)
            
            if dataset == 'IAM':
                page = '-'.join(page.split('-')[:2])
            page_list.append([page]*num_of_retrievals)
            
            bbox_list.append(bboxes)
            score_list.append(scores)

    suggestions = pl.DataFrame({
        "query": np.hstack(query_list),
        "page": np.hstack(page_list),
        "bbox": np.vstack(bbox_list),
        "syntactic_similarity": np.hstack(score_list)
    })
    
    suggestions = suggestions.with_columns(syntactic_similarity=1-pl.col("syntactic_similarity")/pl.col("syntactic_similarity").max())
    
    return pl.concat([
        suggestions,
        pl.DataFrame({
            "query": queries_without_predictions,
            "page": [None]*len(queries_without_predictions),
            "bbox": [None]*len(queries_without_predictions),
            "syntactic_similarity": [None]*len(queries_without_predictions),
        })
    ])
            




def rerank_by(suggestions, key):
    return suggestions.with_columns( 
        ranking = pl.col(key).rank("ordinal", descending=True)
                             .over("query") - 1
    )



def compute_map_of_reranked_by(key, suggestions, ground_truth, iou_threshold, *, keep_topk=None):
    suggestions = rerank_by(suggestions, key)
    return compute_map(suggestions, ground_truth, iou_threshold, keep_topk=keep_topk)
   
   
def compute_map_of_verbatim_kws(suggestions, ground_truth, iou_threshold, *, keep_topk=None):
    return compute_map_of_reranked_by(
        "syntactic_similarity",
        suggestions, 
        ground_truth, 
        iou_threshold, 
        keep_topk=keep_topk
    )
    
    
def compute_map_of_verbatim_kws_with_semantic_cutoff(
    suggestions, 
    ground_truth,
    iou_threshold,
    *, 
    semantic_key,
    cutoff_threshold,
    keep_topk=None
):
    return compute_map_of_reranked_by(
        "syntactic_similarity",
        suggestions.filter( pl.col(semantic_key) > cutoff_threshold ), 
        ground_truth, 
        iou_threshold, 
        keep_topk=keep_topk
    )
 

def compute_map_of_combined_verbatim_and_semantic_kws(
    suggestions, 
    ground_truth, 
    iou_threshold,
    *,
    syntactic_key="syntactic_similarity",
    semantic_key,
    weighting_factor, 
    cutoff_threshold=0,
    keep_topk=None
):
    w = weighting_factor
    return compute_map_of_reranked_by(
        "combined_scores",
        suggestions.with_columns(  combined_scores = w*pl.col(syntactic_key) + (1-w)*pl.col(semantic_key)  ),
        ground_truth, 
        iou_threshold, 
        keep_topk=keep_topk
    )
    




        


        
        
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground-truth-location", type=Path, metavar="folder-with-ground-truth-files", required=True)
    parser.add_argument("--wrn-result-location", type=Path, metavar="location-of-WRN-result-files", required=True)
    parser.add_argument("--segfreekws-result-location", type=Path, metavar="location-of-SegFreeKWS-result-files", required=True)
    parser.add_argument("--save-results-location", type=Path, metavar="path-to-save-results-at", required=True)
    parser.add_argument("--keep-topk", type=int, metavar="only-keep-the-topk-results", nargs="?", default=30)
    parser.add_argument("--avoid-cache", action="store_true", default=None)
    parser.add_argument("--skip-mAP-computation", action="store_true", default=None)
    parser.add_argument("--skip-semantic-filtering", action="store_true", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--only-print-verbatim", action="store_true", default=None)
    parser.add_argument("--rerank-candidates", type=int, nargs="?", default=1000)
    parser.add_argument("--skip-WRN-mAP-computation", action="store_true", default=None)
    parser.add_argument("--skip-SegFreeKWS-mAP-computation", action="store_true", default=None)
    args = parser.parse_args()
    
    dataset = args.dataset
        
    print(f"Loading the ground truth for {dataset}...")
    if dataset == "GW":
        ground_truth, stopwords = load_ground_truth_GW(args.ground_truth_location)
    elif dataset == "IAM":
        ground_truth, stopwords = load_ground_truth_IAM(args.ground_truth_location)
    else:
        raise ValueError("invalid dataset")
    
    print("Loading the WRN results...")
    wrn_results = []
    if args.skip_WRN_mAP_computation is None:
        if dataset == "IAM":
            results_location = args.save_results_location/f"WRN_IAM_results.parquet"
            full_results_location = args.save_results_location/f"full_WRN_IAM_results.parquet"
            if not results_location.exists() or (args.avoid_cache is not None):
                print("---> fresh copy")
                df = load_WRN_bbox_suggestions_as_dataframe(args.wrn_result_location, "IAM")
                df = df.filter(~pl.col('query').is_in(stopwords))
                df.write_parquet(full_results_location)
                df = rerank_by(df, "syntactic_similarity")
                df = df.filter(pl.col("ranking")<args.rerank_candidates)
                df.write_parquet(results_location)
            else:
                print("---> from cache")
                df = pl.read_parquet(results_location)
            wrn_results.append(df)
        else:
            for f in range(4):
                results_location = args.save_results_location/f"WRN_GW{f}_results.parquet"
                if not results_location.exists() or args.avoid_cache:
                    df = load_WRN_bbox_suggestions_as_dataframe(args.wrn_result_location, "GW", f)
                    df.write_parquet(results_location)
                else:
                    df = pl.read_parquet(results_location)
                wrn_results.append(df)
    
    
    print("Loading the SegFreeKWS results...")
    segfreekws_results = []
    if args.skip_SegFreeKWS_mAP_computation is None:
        if dataset == "IAM":
            results_location = args.save_results_location/f"SegFreeKWS_IAM_results.parquet"
            if not results_location.exists() or (args.avoid_cache is not None):
                print("---> fresh copy")
                df = load_SegFreeKWS_bbox_suggestions_as_dataframe(args.segfreekws_result_location, dataset)
                df = df.filter(~pl.col('query').is_in(stopwords))
                df.write_parquet(results_location)
            else:
                print("---> from cache")
                df = pl.read_parquet(results_location)
            segfreekws_results.append(df)
        else:
            for f in range(4):
                results_location = args.save_results_location/f"SegFreeKWS_GW{f}_results.parquet"
                if not results_location.exists() or args.avoid_cache:
                    df = load_SegFreeKWS_bbox_suggestions_as_dataframe(args.segfreekws_result_location, dataset, f)
                    df.write_parquet(results_location)
                else:
                    df = pl.read_parquet(results_location)
                segfreekws_results.append(df)
        
    
    if args.skip_mAP_computation:
        raise SystemExit()
    
    map_scores = []
    
    for kws_method, results in zip(["WRN", "SegFreeKWS"], [wrn_results, segfreekws_results]):
    
        if (kws_method == "WRN" and args.skip_WRN_mAP_computation): 
            print(f"skipping {kws_method} mAP computation")
            continue
        
        if (kws_method == "SegFreeKWS" and args.skip_SegFreeKWS_mAP_computation): 
            print(f"skipping {kws_method} mAP computation")
            continue
        
        print(f"Computing mAP for the verbatim {kws_method} results...")
            
        map_scores.append({   
            "kws_method": kws_method,
            "map@25_over_folds": [
                100*compute_map_of_verbatim_kws(
                    suggestions, 
                    ground_truth, 
                    0.25, 
                    keep_topk = args.keep_topk
                ) for suggestions in results
            ],
            "map@50_over_folds": [
                100*compute_map_of_verbatim_kws(
                    suggestions, 
                    ground_truth, 
                    0.50, 
                    keep_topk = args.keep_topk
                ) for suggestions in results
            ]
        })
        
        if args.only_print_verbatim: 
            continue
        
        semantic_columns = [i for i in results[0].columns if "sem" in i]
        weights = np.linspace(0, 1, num=1+10)
        cutoff_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        
        if args.skip_semantic_filtering is None:
            print(f"Computing mAP for the verbatim {kws_method} results with cutoff based on sematic similarity...")
            map_scores.extend([
                {   
                    "kws_method": kws_method,
                    "semantic_similarity": s,
                    "cutoff_threshold": ct,
                    "map@25_over_folds": [
                        100*compute_map_of_verbatim_kws_with_semantic_cutoff(
                            suggestions, 
                            ground_truth, 
                            0.25, 
                            semantic_key=s,
                            cutoff_threshold=ct,
                            keep_topk = args.keep_topk
                        ) for suggestions in results
                    ],
                    "map@50_over_folds": [
                        100*compute_map_of_verbatim_kws_with_semantic_cutoff(
                            suggestions, 
                            ground_truth, 
                            0.50, 
                            semantic_key=s,
                            cutoff_threshold=ct,
                            keep_topk = args.keep_topk
                        ) for suggestions in results
                    ],
                } for ct in cutoff_levels
                    for s in semantic_columns
            ])


        print(f"Computing mAP for the combined verbatim and semantic {kws_method} results...")
        
        map_scores.extend([
            {   
                "kws_method": kws_method,
                "semantic_similarity": s,
                "weighting_factor": w,
                "map@25_over_folds": [
                    100*compute_map_of_combined_verbatim_and_semantic_kws(
                        suggestions, 
                        ground_truth, 
                        0.25, 
                        semantic_key=s,
                        weighting_factor=w,
                        keep_topk = args.keep_topk
                    ) for suggestions in results
                ],
                "map@50_over_folds": [
                    100*compute_map_of_combined_verbatim_and_semantic_kws(
                        suggestions, 
                        ground_truth, 
                        0.50, 
                        semantic_key=s,
                        weighting_factor=w,
                        keep_topk = args.keep_topk
                    ) for suggestions in results
                ]
            } for w in weights
                for s in semantic_columns
        ])
     
            
    np.save(args.save_results_location/f"map_results_backup.npy", map_scores)    
    map_scores = pl.from_dicts(map_scores)
    
    map_scores = map_scores.with_columns(pl.col("map@25_over_folds").list.mean().alias("map@25"))
    map_scores = map_scores.with_columns(pl.col("map@25_over_folds").list.std().alias("map@25_std"))
    map_scores = map_scores.with_columns(pl.col("map@50_over_folds").list.mean().alias("map@50"))
    map_scores = map_scores.with_columns(pl.col("map@50_over_folds").list.std().alias("map@50_std"))
    
    print(map_scores)
    
    if args.only_print_verbatim:
        raise SystemExit()
    
    map_scores.write_parquet(args.save_results_location/f"map_results.parquet")
    
    map_scores = map_scores.drop("map@25_over_folds", "map@50_over_folds")
   
    if (args.skip_WRN_mAP_computation): 
        map_scores.write_csv(args.save_results_location/f"map_results_{dataset}_onlySegFreeKWS.csv")
    elif (args.skip_SegFreeKWS_mAP_computation): 
        map_scores.write_csv(args.save_results_location/f"map_results_{dataset}_onlyWRN.csv")
    else:     
        map_scores.write_csv(args.save_results_location/f"map_results_{dataset}.csv")
    print(map_scores)
