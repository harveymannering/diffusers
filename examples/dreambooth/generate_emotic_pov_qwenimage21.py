"""Generate POV images for every person in EMOTIC with a trained Qwen-Image 2.1 LoRA.

Each person is annotated as in emotic_test_20: a square crop around their EMOTIC body box resized to 512, the box
outlined in red, and their Keypoint R-CNN pose drawn on top (`--annotation box` leaves the pose out).

    python generate_emotic_pov_qwenimage21.py --lora_path pytorch_lora_weights.safetensors --output_dir out --max_images 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.ops import box_iou
from torchvision.transforms.functional import to_tensor


# COCO keypoint pairs: nose, eyes and ears first, then the body
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)]
SKELETON += [(11, 13), (13, 15), (12, 14), (14, 16)]


def as_list(x):
    """scipy collapses single-element struct arrays; force them back into a list."""
    return list(x) if isinstance(x, (list, np.ndarray)) else [x]


def load_detector(device):
    return torchvision.models.detection.keypointrcnn_resnet50_fpn(weights="COCO_V1").eval().to(device)


@torch.no_grad()
def detect_people(detector, image):
    return detector([to_tensor(image).to(next(detector.parameters()).device)])[0]


def annotate(image, bbox, detections=None, size=512, min_iou=0.5):
    """Crop a square around `bbox`, outline it in red and draw the pose of the detection that best overlaps it.

    Returns the annotated image and the IoU of the drawn detection, or None when no pose was drawn.
    """
    w, h = image.size
    side = min(w, h)
    x1, y1, x2, y2 = bbox
    # Centred on the box, but keeping its top-left corner in view when the box is larger than the crop.
    left = max(min(int(min(max((x1 + x2 - side) / 2, 0), w - side)), int(x1)), 0)
    top = max(min(int(min(max((y1 + y2 - side) / 2, 0), h - side)), int(y1)), 0)

    image = image.copy()
    ImageDraw.Draw(image).rectangle([int(v) for v in bbox], outline="red", width=max(1, round(side / 128)))
    image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)

    if detections is None or len(detections["boxes"]) == 0:
        return image, None
    ious = box_iou(torch.tensor([bbox], dtype=torch.float32, device=detections["boxes"].device), detections["boxes"])[0]
    j = int(ious.argmax())
    if ious[j] < min_iou:
        return image, None

    keypoints = (detections["keypoints"][j, :, :2].cpu().numpy() - [left, top]) * size / side
    visible = (detections["keypoints_scores"][j] > 2).tolist()
    draw = ImageDraw.Draw(image)
    for a, b in SKELETON:
        if visible[a] and visible[b]:
            draw.line([tuple(keypoints[a]), tuple(keypoints[b])], fill=(0, 255, 0), width=3)
    for i, (x, y) in enumerate(keypoints):
        if visible[i]:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 255, 0) if i < 5 else (0, 255, 255))
    return image, float(ious[j])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emotic_root", default="/scratch/hgm1g14/EMOTIC_PAMI")
    parser.add_argument("--pretrained_model_name_or_path", default="Qwen/Qwen-Image-2.1")
    parser.add_argument("--lora_path", required=True, help="LoRA .safetensors file, or the folder holding it.")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument(
        "--prompt",
        default="Make the camera the POV perspective of the person in the highlighted person, high quality photo",
    )
    parser.add_argument("--negative_prompt", default=None, help="Only used when --true_cfg_scale > 1.")
    parser.add_argument("--true_cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_images_per_person", type=int, default=1, help="Image i uses seed `--seed + i`.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=None, help="Stop after generating this many images.")
    parser.add_argument("--annotation", choices=["pose", "box"], default="pose")
    parser.add_argument("--min_iou", type=float, default=0.5, help="Below this, the person gets the red box only.")
    parser.add_argument("--shard", type=int, default=0, help="Process every `--num_shards`-th image, from this one.")
    parser.add_argument("--num_shards", type=int, default=1)
    return parser.parse_args()


def main(args):
    from diffusers import QwenImage21Pipeline

    root = Path(args.emotic_root)
    ann = sio.loadmat(root / "Annotations" / "Annotations" / "Annotations.mat", simplify_cells=True)
    # The archive unpacks to either emotic/<source> or emotic/emotic/<source>
    image_root = root / "emotic" / "emotic" if (root / "emotic" / "emotic").is_dir() else root / "emotic"
    records = [(split, rec) for split in args.splits.split(",") for rec in ann[split]][args.shard :: args.num_shards]

    pipe = QwenImage21Pipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    pipe.load_lora_weights(args.lora_path)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    detector = load_detector("cuda") if args.annotation == "pose" else None

    out = Path(args.output_dir)
    for split in args.splits.split(","):
        (out / split).mkdir(parents=True, exist_ok=True)
    metadata = open(out / f"metadata_shard{args.shard}.jsonl", "a")
    generated = 0
    for n, (split, rec) in enumerate(records):
        people = as_list(rec["person"])
        stems = [out / split / f"{Path(rec['filename']).stem}_person{j}" for j in range(len(people))]
        last = args.num_images_per_person - 1
        if all(Path(f"{s}_pov{last}.png").exists() for s in stems):
            continue  # finished by an earlier run
        image = Image.open(image_root / rec["folder"] / rec["filename"]).convert("RGB")
        detections = detect_people(detector, image) if detector else None

        for j, (person, stem) in enumerate(zip(people, stems)):
            bbox = [float(v) for v in person["body_bbox"]]
            annotated, iou = annotate(image, bbox, detections, min_iou=args.min_iou)
            annotated.save(f"{stem}_annotated.png")
            record = {"split": split, "folder": rec["folder"], "filename": rec["filename"], "person_idx": j}
            record.update(body_bbox=bbox, pose_iou=iou, prompt=args.prompt, seed=args.seed)
            record.update(num_images=args.num_images_per_person, true_cfg_scale=args.true_cfg_scale)
            record.update(num_inference_steps=args.num_inference_steps, lora_scale=args.lora_scale)
            metadata.write(json.dumps(record) + "\n")
            metadata.flush()
            for i in range(args.num_images_per_person):
                pov = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    image=annotated,
                    true_cfg_scale=args.true_cfg_scale,
                    num_inference_steps=args.num_inference_steps,
                    output_resolution=args.resolution,
                    generator=torch.Generator("cuda").manual_seed(args.seed + i),
                    attention_kwargs={"scale": args.lora_scale},
                ).images[0]
                pov.convert("RGB").save(f"{stem}_pov{i}.png")
                generated += 1
                if args.max_images and generated >= args.max_images:
                    return
        print(f"[shard {args.shard}] {n + 1}/{len(records)} images, {generated} POVs generated", flush=True)


if __name__ == "__main__":
    main(parse_args())
