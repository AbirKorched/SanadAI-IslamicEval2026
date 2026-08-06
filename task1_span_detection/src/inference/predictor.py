import torch

from .aggregation import aggregate_word_prediction


def predict_text(
    text,
    model,
    tokenizer,
    device,
    label2id,
    thresholds
):


    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        return_offsets_mapping=True
    )


    word_ids = inputs.word_ids(0)


    model_inputs={
        k:v.to(device)
        for k,v in inputs.items()
        if k!="offset_mapping"
    }


    with torch.no_grad():

        logits=model(**model_inputs).logits[0]


    probs=torch.softmax(
        logits,
        dim=-1
    )


    words={}


    for token_idx,word_id in enumerate(word_ids):

        if word_id is None:
            continue


        if word_id not in words:

            span=inputs.word_to_chars(
                0,
                word_id
            )


            words[word_id]={
                "start":span.start,
                "end":span.end,
                "probs":[]
            }


        words[word_id]["probs"].append(
            probs[token_idx].cpu()
        )



    predictions=[]


    for _,info in words.items():

        entity,score = aggregate_word_prediction(
            info["probs"],
            label2id
        )


        if entity!="O":

            if score < thresholds[entity]:
                entity="O"


        predictions.append(
            {
                "start":info["start"],
                "end":info["end"],
                "entity":entity,
                "score":score
            }
        )


    return predictions