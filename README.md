# Multi-Target Unsupervised Domain Adaptation for Semantic Segmentation without External Data

This is the official PyTorch implementation of the domain adaptation method in our paper [Multi-Target Unsupervised Domain Adaptation for Semantic Segmentation without External Data](https://arxiv.org/abs/2405.06502).

## Training the Multi-Target Style Transfer Network (MT-STN) 
```
$ CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=4 --node_rank=0 train_MT-STN.py
```
- Use tensorboard to monitor the training process:
```
$ tensorboard --logdir ./
```

## Style Transfer 
```
$ CUDA_VISIBLE_DEVICES=0 python gen_style_trans.py
```

## Training MT-KD
- Synthetic-to-real adaptation with 1 target domain: (~28GB memory)
```
$ CUDA_VISIBLE_DEVICES=0 python train_MT-KD-1T-from-GTA.py
```
- Synthetic-to-real adaptation with 2 target domains: (~39GB memory)
```
$ CUDA_VISIBLE_DEVICES=0 python train_MT-KD-2T-from-GTA.py
```
- Synthetic-to-real adaptation with 3 target domains: (~50GB memory A100 80G)
```
$ CUDA_VISIBLE_DEVICES=0 python train_MT-KD-3T-from-GTA.py
```
- Real-to-real adaptation with 1 target domain: (~22GB memory) 
```
$ CUDA_VISIBLE_DEVICES=0 python train_MT-KD-1T-from-Cityscapes.py
```
- Real-to-real adaptation with 2 target domains: (~33GB memory)
```
$ CUDA_VISIBLE_DEVICES=0 python train_MT-KD-2T-from-Cityscapes.py
```

## Training UT-KD
```
$ CUDA_VISIBLE_DEVICES=0 python train_UT-KD.py
```

## Evaluation
```
$ CUDA_VISIBLE_DEVICES=0 python evaluation.py
```

[Pretrained models](https://drive.google.com/file/d/1ng2P-V3Adh1eGGwrJWxhB87Ch3JqJl5m/view?usp=sharing)

## License
This repo is distributed under [MIT License](https://github.com/YonghaoXu/UT-KD/blob/main/LICENSE). The code can be used for academic purposes only.