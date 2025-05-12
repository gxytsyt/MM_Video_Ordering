import os

import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
import tqdm
from decord import VideoReader, cpu
from transformers import CLIPVisionModel, CLIPImageProcessor
from transformers import CLIPProcessor, CLIPModel
import json


def load_video(vis_path, num_frm=100):
    vr = VideoReader(vis_path, ctx=cpu(0))
    total_frame_num = len(vr)
    total_num_frm = min(total_frame_num, num_frm)
    frame_idx = get_seq_frames(total_frame_num, total_num_frm)
    img_array = vr.get_batch(frame_idx).asnumpy()  # (n_clips*num_frm, H, W, 3)

    a, H, W, _ = img_array.shape
    # print(H)
    # print(W)
    # input('s')
    h, w = 224, 224
    if img_array.shape[-3] != h or img_array.shape[-2] != w:
        img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
        img_array = torch.nn.functional.interpolate(img_array, size=(h, w))
        img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()
    img_array = img_array.reshape((1, total_num_frm, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))

    clip_imgs = []
    for j in range(total_num_frm):
        clip_imgs.append(Image.fromarray(img_array[0, j]))

    return clip_imgs


def get_seq_frames(total_num_frames, desired_num_frames):
    seg_size = float(total_num_frames - 1) / desired_num_frames
    seq = []
    for i in range(desired_num_frames):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        seq.append((start + end) // 2)

    return seq


def get_spatio_temporal_features(features, num_temporal_tokens=40):
    t, s, c = features.shape

    temporal_tokens = np.mean(features, axis=1)
    padding_size = num_temporal_tokens - t
    if padding_size > 0:
        temporal_tokens = np.pad(temporal_tokens, ((0, padding_size), (0, 0)), mode='constant')

    spatial_tokens = np.mean(features, axis=0)
    sp_features = np.concatenate([temporal_tokens, spatial_tokens], axis=0)

    return sp_features, temporal_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--clip_feat_path", default='./Wikihow_video_data/clip_features', help="The output dir to save the features in.")
    parser.add_argument("--infer_batch", required=False, type=int, default=20,
                        help="Number of frames/images to perform batch inference.")

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    os.makedirs(clip_feat_path, exist_ok=True)

    processor = CLIPProcessor.from_pretrained("clip-vit-base-patch16")
    vision_tower = CLIPModel.from_pretrained("clip-vit-base-patch16").cuda()

    vision_tower.eval()
    json_file_video = 'Wikihow_video_data/pure_video_data.json'
    all_videos_path = []
    with open(json_file_video, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        for item in tqdm.tqdm(datas):
            steps_all = item['section']['steps_all']
            for step_i, step in enumerate(steps_all):
                video_url = step['video_url']
                if os.path.isfile('pure_videos' + video_url):
                    all_videos_path.append('pure_videos' + video_url)

    all_videos = all_videos_path

    video_clip_features = {}
    counter = 0
    for video_name in tqdm.tqdm(all_videos):
        # video_path = f"{video_dir_path}/{video_name}"
        video_path = video_name
        inner_name = video_path.split("/", 1)[1]
        inner_folder = inner_name.rsplit("/", 1)[0]

        if not os.path.exists(clip_feat_path + '/' + inner_folder):
            os.makedirs(clip_feat_path + '/' + inner_folder)

        if os.path.exists(f"{clip_feat_path}/{inner_name}.pkl"):  # Check if the file is already processed
            continue
        try:
            # video = load_video(video_path, num_frm=40)
            video = load_video(video_path, num_frm=16)

            inputs = processor(images=video, return_tensors="pt", padding=True)
            inputs = {key: value.cuda() for key, value in inputs.items()}
            video_tensor = inputs
            image_forward_outs = vision_tower.get_image_features(**video_tensor)

            # print('image_forward_outs', image_forward_outs.shape)
            # input('sss')
            video_clip_features[inner_name] = image_forward_outs

            counter += 1

        except Exception as e:
            print(e)
            print(f"Can't process {video_path}")

        if counter % 3 == 0:
            for key in video_clip_features.keys():
                features = video_clip_features[key]
                with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
                    pickle.dump(features, f)
            video_clip_features = {}

    for key in video_clip_features.keys():
        features = video_clip_features[key]
        with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
            pickle.dump(features, f)


if __name__ == "__main__":
    main()
