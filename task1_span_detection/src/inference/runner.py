import pandas as pd
from tqdm import tqdm


from src.inference.predictor import predict_text
from src.postprocess.spans import merge_word_spans



def run_inference(
    df,
    model,
    tokenizer,
    device,
    label2id,
    thresholds
):

    rows=[]


    for _,row in tqdm(df.iterrows(), total=len(df)):


        response_id=row["id"]
        text=row["generated_answer"]


        if not text.strip():

            rows.append({
                "Response_ID":response_id,
                "Segment_Type":"NoAnnotation",
                "Span_Start":"-",
                "Span_End":"-",
                "Score":0.0
            })

            continue



        words=predict_text(
            text,
            model,
            tokenizer,
            device,
            label2id,
            thresholds
        )


        spans=merge_word_spans(words, text)



        if not spans:

            rows.append({
                "Response_ID":response_id,
                "Segment_Type":"NoAnnotation",
                "Span_Start":"-",
                "Span_End":"-",
                "Score":0.0
            })

            continue



        for s in spans:

            rows.append({
                "Response_ID":response_id,
                "Segment_Type":s["entity"],
                "Span_Start":s["start"],
                "Span_End":s["end"],
                "Score":round(
                    s["score"],
                    6
                )
            })


    return pd.DataFrame(rows)