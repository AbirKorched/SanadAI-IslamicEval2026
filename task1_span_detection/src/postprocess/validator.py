def validate(
    df,
    min_length,
    max_source_distance
):

    output = []


    for _, group in df.groupby(
        "Response_ID",
        sort=False
    ):

        group = (
            group
            .sort_values("Span_Start")
        )


        kept = []


        for _, row in group.iterrows():
            if row.Segment_Type == "NoAnnotation":
                kept.append(row)
                continue

            entity = row.Segment_Type
            length = (
                row.Span_End
                -
                row.Span_Start
            )


            if length < min_length[entity]:
                continue


            if entity == "claimed_source":

                valid = False


                for prev in reversed(kept):

                    if prev.Segment_Type not in {
                        "ayah",
                        "matn"
                    }:
                        continue


                    distance = (
                        row.Span_Start
                        -
                        prev.Span_End
                    )


                    if (
                        distance >=0
                        and distance <= max_source_distance
                    ):
                        valid = True
                        break


                if not valid:
                    continue


            kept.append(row)


        output.extend(kept)


    return df.__class__(output)