import pandas as pd 
import numpy as np 
import io
from Dataset.Registry import dataset_registry

def DescriptiveAnalysis(state):
    """
    This node performs descriptive analysis on the dataset.
    It returns the descriptive results in the state.

    output: state["descriptive_results"] = {"info": info, "desc": desc, "shape": shape, "target_dtype": target_dtype, "num_cols": num_cols, "cat_cols": cat_cols}
    """
    
    dataset_id = state["dataset_id"]

    df = dataset_registry.get(dataset_id)

    # Convert to serializable formats
    info_buffer = io.StringIO() #for memory checkpointer storing df.info(), df.description() as they are dataframe obj

    df.info(buf=info_buffer)
    info_str = info_buffer.getvalue()
    desc_dict = df.describe().to_dict()
    shape = df.shape
    target_dtype = str(df[state["target_column"]].dtype)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if state["target_column"] in num_cols: num_cols.remove(state["target_column"]) 
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if state["target_column"] in cat_cols: cat_cols.remove(state["target_column"])

    return {
        "descriptive_results": {
            "dataset_id": dataset_id,
            "target_column": state["target_column"],
            "info": info_str,  # String instead of None
            "desc": desc_dict,  # Dict instead of DataFrame
            "shape": shape,
            "target_dtype": target_dtype  # String instead of dtype object
            # num_cols and cat_cols removed - now only at top level
        },
        "num_columns": num_cols,  # Top-level only
        "cat_columns": cat_cols   # Top-level only
    }


    
