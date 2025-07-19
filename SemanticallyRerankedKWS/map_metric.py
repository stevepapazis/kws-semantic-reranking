import numpy as np
import polars as pl
import torch
import torchvision.ops as tvops


    
def compute_relevance_in_page(bboxes, gt_bboxes, iou_threshold): 
    if gt_bboxes is None:
        return np.zeros(len(bboxes), dtype=np.float32) 
    if bboxes is None:
        return np.zeros(0, dtype=np.float32)

    num_bboxes = len(bboxes)
    relevant_instances = np.zeros(num_bboxes, dtype=np.float32)
   
    num_gt_bboxes = len(gt_bboxes)  
    seen_gt_bboxes = np.zeros(num_gt_bboxes, dtype=np.float32)
    
    i = 0
    while i < num_bboxes and seen_gt_bboxes.sum() < num_gt_bboxes:
        iou, reverse_index = tvops.box_iou(
            torch.Tensor(bboxes[np.newaxis, i]),
            torch.Tensor(gt_bboxes)
        ).squeeze(0).sort(descending=True)
        
        j = 0
        while j < num_gt_bboxes and iou[j] > iou_threshold:
            gt_j = reverse_index[j]
            if not seen_gt_bboxes[gt_j]:
                seen_gt_bboxes[gt_j] = 1
                relevant_instances[i] = 1
                break
            j += 1
        i += 1
    
    assert relevant_instances.sum() <= num_gt_bboxes
    
    return relevant_instances
    
    
        
def compute_average_precision(relevant_instances, actual_number_of_relevant_instances):
    assert relevant_instances.ndim == 1 and relevant_instances.sum() <= actual_number_of_relevant_instances
    
    relevant_instances_at_k_retrievals = np.cumsum(relevant_instances, dtype=float)
    number_of_retrievals = 1 + np.arange(len(relevant_instances))
    precision_at_k_retrievals = relevant_instances_at_k_retrievals / number_of_retrievals
    
    if actual_number_of_relevant_instances>0:
        average_precision = (precision_at_k_retrievals * relevant_instances).sum() / actual_number_of_relevant_instances
    else:
        average_precision = 0
    
    return average_precision



def compute_map(suggestion_df, ground_truth_df, iou_threshold, *, keep_topk):
    average_precision_per_query = []
    
    # queries for which the system has no retrievals
    average_precision_per_query.extend(
        [0.0]*len(suggestion_df.filter(pl.col('bbox').is_null()))
    )
    
    suggestion_df = (
        suggestion_df.filter(pl.col("ranking") < keep_topk)
                     .sort("ranking")
                     .group_by(["query", "page"])
                     .all()
    )
    
    ground_truth_df = ground_truth_df.filter(
        pl.col("query").is_in(suggestion_df["query"]),
        pl.col("page").is_in(suggestion_df["page"]),
    ).group_by(["query", "page"]).all()
        
    joined_df = suggestion_df.join(
        ground_truth_df,
        on=["query","page"], 
        how="full",
        coalesce=True
    )
    
    for _, query_df in joined_df.group_by("query"):
        if query_df["bbox"].is_null().all():
            print(query_df['query'].unique().item())
            continue
                
        relevant_instances = np.zeros(
            query_df.select(pl.col('bbox').list.len().sum()).item(), 
            dtype=np.float32
        )
        
        for gt_bboxes, bboxes, ranking in query_df["gt_bbox", "bbox", "ranking"].iter_rows():
            if bboxes is None or gt_bboxes is None:   
                continue

            relevant_instances[ranking] = compute_relevance_in_page(         
                np.vstack(bboxes),
                np.vstack(gt_bboxes),
                iou_threshold
            )
            
        number_of_gt_bboxes = (
            query_df.select( pl.col("gt_bbox").list.len() )
                    .sum()
                    .item()
        )
                
        number_of_gt_bboxes = min(keep_topk, number_of_gt_bboxes)
        
        if not (relevant_instances.sum() <= number_of_gt_bboxes):
            print(query_df)
            print(relevant_instances)
            print(number_of_gt_bboxes)
            
        assert relevant_instances.sum() <= number_of_gt_bboxes        
        
        try:
            average_precision = compute_average_precision(relevant_instances, number_of_gt_bboxes)
        except AssertionError:
            print(query_df)
            print(relevant_instances)
            print(number_of_gt_bboxes)
            raise
        average_precision_per_query.append(average_precision)

    return np.mean(average_precision_per_query)
    
