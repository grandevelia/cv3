from PIL import Image, ImageDraw
import json

with open('../logs/voc_fcos_2022-12-21 10:15:23/eval_results.json') as f:
    d = json.load(f)

n = 100

cls_names = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

out_dir = "../test"
im_dir = '../data/VOCdevkit/VOC2007/JPEGImages'

def draw_box(curr_dat):
    im_id = str(curr_dat['image_id'])
    while len(im_id) < 6:
        im_id = '0' + im_id
    box = curr_dat['bbox']
    cat = cls_names[int(curr_dat['category_id']-1)]
    im = Image.open(f"{im_dir}/{im_id}.jpg")
    draw = ImageDraw.Draw(im)
    draw.line((box[0], box[1], box[0], box[3]), fill=(255, 0, 0), width=3)
    draw.line((box[0], box[1], box[2], box[1]), fill=(255, 0, 0), width=3)
    draw.line((box[2], box[1], box[2], box[3]), fill=(255, 0, 0), width=3)
    draw.line((box[2], box[3], box[0], box[3]), fill=(255, 0, 0), width=3)
    im.save(f"{out_dir}/{im_id}_{cat}.jpg")

for i, curr_dat in enumerate(d):
    draw_box(curr_dat)
    if i > n:
        break


