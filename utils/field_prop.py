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
        The rate map, which is the output of the second CA1 layer.
    parameter : float, optional
        The threshold for field identification. The default is 0.4.
    split_thre : float, optional
        The threshold to split two close fields. It controls the saddle-to-higher peak ratio, in 
        terms of their rate. The default is 0.2.

    Returns
    -------
    dict
        The field dictionary.
    """

    # The indexes of those bins with rate higher than threshold will be extracted.
    all_fields = np.where(rate_map >= parameter)[0]+1
    # Areas/bins that have been identified as part of a field/fields, which will not considered further
    searched_set = np.array([], dtype=np.int64)
    place_fields = {}

    while len(np.setdiff1d(all_fields, searched_set))!=0:
        # Get those bins that have not been identified as part of a field
        diff = np.setdiff1d(all_fields, searched_set)
        # Get the index of the bin with the highest rate in the remaining bins
        diff_idx = np.argmax(rate_map[diff-1])
        # Based on this index to get the subfield, which is returned in the form of a numpy array.
        subfield = _field(diff = diff, diff_idx = diff_idx, split_thre=split_thre, rate_map = rate_map)

        place_fields[diff[diff_idx]] = subfield # Using a dict object to contain all the identified fields,
        # Key: Value  ->  field center position: all the bins of this field
        # Bins of the newly identified subfield are added to the searched set.
        searched_set = np.concatenate([searched_set, subfield])

    return place_fields
               
def _field(
    diff: list | np.ndarray, 
    diff_idx: int, 
    split_thre: float = 0.2,
    rate_map: np.ndarray | None = None
) -> np.ndarray:
    # Using a recursive algorithm (Breadth-First Search) to generate fields based on 
    # the center and all candidate bins.
    
    point = diff[diff_idx]
    peak_rate = rate_map[point-1]
    MaxStep = 300 # Maximum number of steps
    step = 0 # Current step
    Area = [point] # Area records all the included bins, in the sense of all steps
    StepExpand = {0: [point]} # StepExpand records the newly included bins in each step
    
    for i in range(1, MaxStep+1): 
        StepExpand[i] = []
        for k in StepExpand[i-1]:
            for j in [k+1, k-1]:
                # If 1) the adjacent bin is not in the candidate bin list
                # and 2) the rate of the bin indicates that it is a saddle point
                # (because a transition of gradient from rate descent to rate ascent is detected)
                if j in diff and j not in Area and (rate_map[j-1] < rate_map[k-1] or 
                                                    rate_map[j-1] >= split_thre*peak_rate):
                    # If the criteria are met, the bin will be included in the field
                    StepExpand[step+1].append(j)
                    Area.append(j)
        
        # If no bin is included in the field, it demonstrates that the field boundary
        # has been reached. The search will be terminated.
        if len(StepExpand[i]) == 0:
            break
    
    return np.array(Area, dtype=int)


def get_place_fields(Y: np.ndarray, parameter: float = 0.4, split_thre: float = 0.2) -> list[dict]:
    """
    Get place fields for all simulated neuron.
    
    Parameters
    ----------
    Y : np.ndarray, shape (n_neuron, n_frame)
        The output of the second CA1 layer.

    Returns
    -------
    list[dict]
        A list for all simulated neurons, which contains n_neuron dict objects
        that contains the field information (dict) for each neuron.
    """
    place_field_all = []
    
    for i in range(Y.shape[0]):
        place_field_all.append(_get_fields(rate_map = Y[i, :], parameter=parameter, split_thre=split_thre))
    
    return place_field_all

def get_field_num(input: list[dict] | np.ndarray) -> np.ndarray:
    """
    Count the number of fields for each simulated neuron
    
    Parameters
    ----------
    input : list[dict] | np.ndarray
        The input can be either a raw output of the second CA1 layer
        or a list of dict objects that contains the field information
        (dict) for each neuron.
    
    Returns
    -------
    np.ndarray, (n_neuron, ), dtype = int
        A numpy array that contains the number of fields for each neuron.
    """
    if isinstance(input, list):
        return np.array([len(i.keys()) for i in input])
    elif isinstance(input, np.ndarray):
        place_fields = get_place_fields(input)
        return np.array([len(i) for i in place_fields])
    else:
        raise ValueError(f"Wrong input type: {type(input)}. Accepted types: list[dict] | np.ndarray")
    
def get_field_size(input: list[dict] | np.ndarray) -> np.ndarray:
    """
    Sizes of fields for each simulated neuron
    
    Parameters
    ----------
    input : list[dict] | np.ndarray
        The input can be either a raw output of the second CA1 layer
        or a list of dict objects that contains the field information
        (dict) for each neuron.
    
    Returns
    -------
    np.ndarray, (n_neuron, ), dtype = int
        A numpy array that contains the sizes of fields for each neuron.
    """
    if isinstance(input, list):
        pass
    elif isinstance(input, np.ndarray):
        input = get_place_fields(input)
    else:
        raise ValueError(f"Wrong input type: {type(input)}. Accepted types: list[dict] | np.ndarray")
    
    sizes = []
    for i in range(len(input)):
        for k in input[i].keys():
            sizes.append(len(input[i][k]))
    return np.array(sizes, int)