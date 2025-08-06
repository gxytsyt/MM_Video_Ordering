# Video Ordering

## 1. Requirements

* **Python version**: 3.7.0
* **PyTorch version**: 1.13.1+cu117
* **TorchVision version**: 0.14.1+cu117
* **Transformers version**: 4.30.2

**Additional Downloads**:

* Download `pytorch_model.bin` for `bart-large` from Huggingface Transformers:
  [Huggingface Transformers](https://github.com/huggingface/transformers)
  Place the downloaded file in `facebook/bart_large/`.

## 2. Data Download

1. Download the text descriptions and corresponding video links from https://drive.google.com/file/d/1BsOQR4jvjbYD-GxVj7GwKCmlLr8uqFyx/view?usp=drive_link
2. Save the downloaded JSON file into `./Wikihow_video_data`.
3. Download the raw videos using ./download/download_videos.py. (The raw video data will be made publicly available upon acceptance.)
4. Use `preprocess_video/feature_from_video.py` to extract video features from the data. This step converts raw video data into a format that can be used for model training.

## 3. Finetune

1. Modify the following paths in the `run_berson_bart.sh` script:

   * `--data_dir`: Path to the processed video dataset.
   * `--model_name_or_path`: Path to the pretrained BART model. Set as `facebook/bart_large` for fine-tuning only.

2. Run the script to fine-tune the model:

   ```bash
   bash run_train.sh
   ```
