def _pick_last_field_as_prompt(text: str):
    """
    Return the last segment of `text` after splitting by common delimiters.
    Supported delimiters: '|', ';', ','.
    If none of them appear, return the stripped text.
    """
    if not isinstance(text, str):
        return None
    for sep in ["|", ";", ","]:
        if sep in text:
            text = text.split(sep)[-1].strip()
    text = text.strip()
    return text if text else None


def extract_edit_prompt(record: dict):
    """
    Build the edit prompt from the record.
    Priority:
      1) 'perturb_prompt' / 'edit_prompt' / 'perturbation_prompt'
      2) Last segment of 'question'
      3) Concatenate target_color (or color) with the first object
    Returns: (prompt, negative_prompt)
    """
    negative = "low quality, blurry"

    # 1) Explicit perturbation prompt fields
    for k in ["perturb_prompt", "edit_prompt", "perturbation_prompt"]:
        if k in record:
            cand = _pick_last_field_as_prompt(record[k])
            if cand:
                return cand, negative

    # 2) Derive from 'question'
    if "question" in record:
        cand = _pick_last_field_as_prompt(record["question"])
        if cand:
            return cand, negative

    # 3) Fallback: "<color> <object>"
    color = None
    if "target_color" in record:
        color = str(record["target_color"]).strip()
    elif "color" in record:
        color = str(record["color"]).strip()

    obj = None
    if "involved_objects" in record and isinstance(record["involved_objects"], (list, tuple)) and record["involved_objects"]:
        obj = str(record["involved_objects"][0]).strip()
    elif "object" in record:
        obj = str(record["object"]).strip()

    if color and obj:
        return f"{color} {obj}", negative

    raise KeyError("No valid prompt fields found in record.")


def write_prompt_log(path: str, qid: str, prompt: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"Question ID: {qid}\n")
        f.write(f"Prompt: {prompt}\n\n")
