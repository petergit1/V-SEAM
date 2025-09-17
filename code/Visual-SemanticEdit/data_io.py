import os

def iter_questions(data):
    # Iterate over question dictionary {qid: record}
    for qid, rec in data.items():
        yield str(qid), rec


def resolve_paths(image_dir, mask_dir, output_dir, qid, image_id):
    image_path = os.path.join(image_dir, f"{image_id}.jpg")
    mask_path = os.path.join(mask_dir, qid, f"{qid}.png")
    out_path = os.path.join(output_dir, qid, f"{qid}_corrupt.png")
    return image_path, mask_path, out_path


def ensure_parent(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
