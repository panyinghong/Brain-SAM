# Brain-SAM  

## Installation 

- Create a virtual environment: `conda create -n brainsam python=3.10 -y` and `conda activate brainsam` 
- Install [PyTorch](https://pytorch.org/get-started/locally/): `pip install torch==2.3.1` (Linux CUDA 12.4)
- cd Brain-SAM
- `pip install -e .`

## datasets
```python
npz = np.load('path to/brain_tumor.npz', allow_pickle=True)
imgs = npz['imgs'] # (D, W, H), [0, 255]
gts = npz['gts'] # (D, W, H), 3D tumor ground truth mask. It will be not available in the testing set
```

## Training Brain-SAM
```bash
nohup env CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --data_prefix Your_data_prefix --snapshot_path your/work_dir --max_epoch 200 --rand_crop_size 128 --data_name your_data_name --lr 6e-5 -bs 3 > train.log 2>&1 &
```

## Inference
```bash
python -m torch.distributed.run --nproc_per_node=2 --master_port=29857 test.py
```

