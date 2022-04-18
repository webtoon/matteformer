# MatteFormer

---

This repository includes the official project of MatteFormer, presented in our paper:
[MatteFormer: Transformer-Based Image Matting via Prior-Tokens](https://arxiv.org/abs/2203.15662) [CVPR 22]

![Exp](assets/exp2.png)

In this paper, we propose a transformer-based image matting model called MatteFormer, which takes full advantage of trimap information in the transformer block. Our method first introduces a prior-token which is a global representation of each trimap region (e.g. foreground, background and unknown). These prior-tokens are used as global priors and participate in the self-attention mechanism of each block. Each stage of the encoder is composed of PAST (Prior-Attentive Swin Transformer) block, which is based on the Swin Transformer block, but differs in a couple of aspects: 1) It has PA-WSA (Prior-Attentive Window Self-Attention) layer, performing self-attention not only with spatial-tokens but also with prior-tokens. 2) It has prior-memory which saves prior-tokens accumulatively from the previous blocks and transfers them to the next block. We evaluate our MatteFormer on the commonly used image matting datasets: Composition-1k and Distinctions-646. Experiment results show that our proposed method achieves state-of-the-art performance with a large margin.

---

### Requirements
The codes are tested in the following environment:
- python 3.8
- pytorch 1.10.1
- CUDA 10.2 & CuDNN 8

### Performances
| Models | SAD | MSE (x10^(-3) | Grad | Conn | Link | 
|:---:|:---:|:---:|:---:|:---:|:---:|
MatteFormer | 23.80 | 4.03 | 8.68 | 18.90 | [model](https://drive.google.com/file/d/1AU7uM1dtYjEhtOa_9OGfoQUE-tmW9mX5/view?usp=sharing) |

---

### Data Preparation
1] Get DIM dataset on [Deep Image Matting](https://sites.google.com/view/deepimagematting).

2] For DIM dataset preparation, please refer to [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting).
- For Training, merge 'Adobe-licensed images' and 'Other' folder to use all 431 foregrounds and alphas
- For Testing, use 'Composition_code.py' and 'copy_testing_alpha.sh' in GCA-Matting.

3] For background images, Download dataset on [PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](https://cocodataset.org/#home).

***If you want to download prepared test set directly : [download link](https://drive.google.com/file/d/1fS-uh2Fi0APygd0NPjqfT7jCwUu_a_Xu/view?usp=sharing)** 

### Testing on Composition-1k dataset
```
pip3 install -r requirements.txt
```

1] Run inference code (the predicted alpha will be save to **./predDIM/pred_alpha** by default)

```
python3 infer.py
```

2] Evaluate the results by the official evaluation MATLAB code **./DIM_evaluation_code/evaluate.m** (provided by [Deep Image Matting](https://sites.google.com/view/deepimagematting))

3] You can also check out the evaluation result simplicity with the python code (un-official) 
```
python3 evaluate.py
```

### Training on Composition-1k dataset
1] You can get (imagenet pretrained) swin-transformer tiny model (**'swin_tiny_patch4_window7_224.pth'**) on [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

2] modify "config/MatteFormer_Composition1k.toml"

3] run main.py
```
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 main.py
```

---

### Citation
If you find this work or code useful for your research, please use the following BibTex entry:
```
@article{park2022matteformer,
  title={MatteFormer: Transformer-Based Image Matting via Prior-Tokens},
  author={Park, GyuTae and Son, SungJoon and Yoo, JaeYoung and Kim, SeHo and Kwak, Nojun},
  journal={arXiv preprint arXiv:2203.15662},
  year={2022}
}
```


### Acknowledgment
- Our Codes are mainly originated from [MG-Matting](https://github.com/yucornetto/MGMatting)
- Also, we build our codes with reference as [GCA-Matting](https://github.com/Yaoyi-Li/GCA-Matting) and [Swin Transformer for Semantic Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation)


### License
MatteFormer is licensed under Apache-2.0, except utils/logger.py which is adopted from https://github.com/JiahuiYu/generative_inpainting under CC BY-NC 4.0.
See [LICENSE](/LICENSE) for the full license text.

```
MatteFormer

Copyright 2022-present NAVER WEBTOON

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

```


![webtoonai](assets/webtoonai.png)
