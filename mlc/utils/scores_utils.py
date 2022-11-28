from mlc.utils.io_utils import get_files_given_a_pattern
import numpy as np
import json
import os
from pathlib import Path

def get_stats_scores(dir_results):
    scores = get_scores(dir_results)
    
    data_out = dict(training_exp=Path(dir_results).stem)
    for scores_by in ["best_2d_iou_score", "best_3d_iou_score"]:
        scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1]['best_iou_valid_score'][scores_by])}        
        list_scores = [sc['best_iou_valid_score'][scores_by] for sc in scores.values()]       
        data_out[f"by_{scores_by}"] = {list(scores.keys())[-1]: scores[list(scores.keys())[-1]]}
        data_out[f"by_{scores_by}"]["stats"]=dict( 
                    std=np.std(list_scores),
                    variance=np.std(list_scores)**2, 
                    median=np.median(list_scores), 
                    mean=np.mean(list_scores), 
                    max=np.max(list_scores), 
                    min=np.min(list_scores), 
                    scores=list_scores,
                    samples=list_scores.__len__())
    
    # scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1]['best_h_valid_score'])}        
    # list_scores = [sc['best_h_valid_score'] for sc in scores.values()]       
    # data_out[f"by_best_h_valid_score"] = {list(scores.keys())[-1]: scores[list(scores.keys())[-1]]}
    # data_out[f"by_best_h_valid_score"]["stats"]=dict( 
    #                 std=np.std(list_scores),
    #                 variance=np.std(list_scores)**2, 
    #                 median=np.median(list_scores), 
    #                 mean=np.mean(list_scores), 
    #                 max=np.max(list_scores), 
    #                 min=np.min(list_scores), 
    #                 scores=list_scores,
    #                 samples=list_scores.__len__())
    return data_out
    
def get_scores(training_dir):
    list_json_fn  = get_files_given_a_pattern(training_dir, "best_score.json", exclude='log')
    if list_json_fn.__len__() == 0:
        assert list_json_fn.__len__() > 0, f"No best_score.json files found at {os.path.join(training_dir)}"
    scores = {}
    for json_fn in list_json_fn:
        exp_label = json_fn.split("/")[-2]
        scores[exp_label] = json.load(open(os.path.join(json_fn, "best_score.json")))
    return scores