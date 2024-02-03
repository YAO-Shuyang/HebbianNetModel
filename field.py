import numpy as np

def _get_fields(
    rate_map: np.ndarray,
    parameter: float = 0.4, 
    split_thre: float = 0.2
) -> dict:
    """
    Get fields from rate map.

    Parameters
    ----------
    rate_map : np.ndarray
        The rate map.
    parameter : float, optional
        The threshold of peak rate. The default is 0.4.
    split_thre : float, optional
        The threshold of peak rate. The default is 0.2.

    Returns
    -------
    dict
        The field dictionary.
    """

    # rate_map should be one without NAN value. Use function clear_NAN(rate_map_all) to process first.
    r_max = np.nanmax(rate_map)
    thre = parameter
    all_fields = np.where(rate_map >= thre)[0]+1
    search_set = np.array([], dtype=np.int64)
    All_field = {}
    

    while len(np.setdiff1d(all_fields, search_set))!=0:
        diff = np.setdiff1d(all_fields, search_set)
        diff_idx = np.argmax(rate_map[diff-1])
        subfield = _field(diff = diff, diff_idx = diff_idx, maze_type = maze_type, 
                          nx = nx, thre=thre, split_thre=split_thre, rate_map = rate_map)

        if IS_QUALIFIED_FIELD:
            submap = rate_map[retain_fields-1]
            peak_idx = np.argmax(submap)
            peak_loc = retain_fields[peak_idx]
            peak = np.max(submap)
        
            All_field[peak_loc] = retain_fields
        

        search_set = np.concatenate([search_set, subfield])
    
    # Sort the fields by their distance to the start point
    res = {}
    for key in sorted(All_field.keys(), key = lambda kv:D[kv-1, 0]):
        res[key] = All_field[key]

    return res
               
def _field(
    diff: list | np.ndarray, 
    diff_idx: int, 
    maze_type: int, 
    nx: int = 48, 
    thre: float = 0.5, 
    split_thre: float = 0.2,
    rate_map: np.ndarray | None = None
) -> np.ndarray:
    # Identify single field from all fields.
    if (maze_type, nx) in maze_graphs.keys():
        graph = maze_graphs[(maze_type, nx)]
    else:
        assert False
    
    point = diff[diff_idx]
    peak_rate = rate_map[point-1]
    MaxStep = 300
    step = 0
    Area = [point]
    StepExpand = {0: [point]}
    while step <= MaxStep:
        StepExpand[step+1] = []
        for k in StepExpand[step]:
            surr = graph[k]
            for j in surr:
                if j in diff and j not in Area and (rate_map[j-1] < rate_map[k-1] or rate_map[j-1] >= split_thre*peak_rate):
                    StepExpand[step+1].append(j)
                    Area.append(j)
        
        # Generate field successfully! 
        if len(StepExpand[step+1]) == 0:
            break
        step += 1
    return np.array(Area, dtype=np.int64)