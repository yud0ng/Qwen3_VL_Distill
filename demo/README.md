# Webcam Spatial Q&A Demo

Live webcam demo for Qwen3-VL spatial reasoning. Point your camera at a scene and ask spatial questions like "how many people are in the frame?" or "where is the red cup?".

## Requirements

- Python 3.10+
- CUDA GPU with ≥6 GB VRAM (runs in bfloat16)
- Webcam

```bash
pip install torch transformers peft opencv-python qwen-vl-utils
```

## Run

```bash
# Default — 2B baseline model
python demo_cv.py

# Specific model by key
python demo_cv.py --model baseline

# List all available models
python demo_cv.py --list-models
```

## Controls (in the OpenCV window)

| Key | Action |
|-----|--------|
| `SPACE` | Capture current webcam frame |
| Type freely | Build your question (after capturing) |
| `ENTER` | Submit question → answer appears on screen |
| `BACKSPACE` | Delete last character |
| `ESC` | Cancel typing / quit |
| `Q` | Quit (when not typing) |

## Configuration

At the top of `demo_cv.py`, set your HuggingFace cache directory:

```python
HF_CACHE = "/your/hf/cache/path"   # or remove to use the default ~/.cache/huggingface
```

## Adding Models

Open `demo_cv.py` and add entries to `MODEL_REGISTRY`. Two types are supported:

### Base model (HF repo or local checkpoint)

```python
"my_model": {
    "type": "base",
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",  # HF repo ID or local path
    "description": "2B Baseline",
},
```

### LoRA adapter

```python
"variant_a": {
    "type": "lora",
    "model_id": "Qwen/Qwen3-VL-2B-Instruct",      # base model to load first
    "adapter_path": "/path/to/adapter_final",       # folder containing adapter_config.json
    "description": "Variant A — fine-tuned on general data",
},
```

The LoRA adapter is merged into the base weights at load time (`merge_and_unload`), so inference speed is the same as a base model.

Then run:

```bash
python demo_cv.py --model variant_a
```

## Notes

- Images are resized to 448×448 before inference to keep VRAM usage low
- Thinking/chain-of-thought is disabled (`enable_thinking=False`) for faster responses
- First inference loads the model (~30s); subsequent queries are fast
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically to reduce OOM fragmentation
