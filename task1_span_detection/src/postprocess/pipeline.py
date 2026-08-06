from .merger import merge_close_spans
from .validator import validate



def postprocess(
    pred,
    cfg
):


    pred = merge_close_spans(
        pred,
        cfg.merge_gap
    )


    pred = validate(
        pred,
        cfg.min_length,
        cfg.max_source_distance
    )


    return pred