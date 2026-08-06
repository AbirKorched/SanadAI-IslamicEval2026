import pandas as pd


def merge_close_spans(
    df,
    merge_gap
):

    df=df.sort_values(
        [
            "Response_ID",
            "Span_Start"
        ]
    )


    merged=[]


    for _,row in df.iterrows():

        row=row.copy()


        if not merged:
            merged.append(row)
            continue


        prev=merged[-1]


        if (
            row.Response_ID != prev.Response_ID
            or
            row.Segment_Type != prev.Segment_Type
        ):
            merged.append(row)
            continue


        gap = (
            row.Span_Start
            -
            prev.Span_End
        )


        if gap <= merge_gap.get(
            row.Segment_Type,
            0
        ):


            prev.Span_End=max(
                prev.Span_End,
                row.Span_End
            )


            prev.Score=max(
                float(prev.Score),
                float(row.Score)
            )


        else:

            merged.append(row)



    return pd.DataFrame(merged)