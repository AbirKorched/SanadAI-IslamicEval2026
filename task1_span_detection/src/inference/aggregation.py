import torch


def get_entity_type(label):

    if label == "O":
        return "O"

    return label.split("-",1)[1]



def aggregate_word_prediction(
    token_probs,
    label2id
):

    probs = torch.stack(token_probs).mean(dim=0)


    entity_scores = {}

    for entity in [
        "ayah",
        "matn",
        "isnad",
        "claimed_source"
    ]:

        entity_scores[entity] = max(
            probs[label2id[f"B-{entity}"]].item(),
            probs[label2id[f"I-{entity}"]].item()
        )


    best_entity = max(
        entity_scores,
        key=entity_scores.get
    )

    best_score = entity_scores[best_entity]


    o_score = probs[label2id["O"]].item()


    if o_score >= best_score:
        return "O", o_score


    return best_entity, best_score