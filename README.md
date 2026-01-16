# Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory

This is the official repository for paper *[Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory](https://arxiv.org/abs/2505.15055)*.

In this paper, we introduce PSN-IRT, a framework based on Item Response Theory (IRT), for a deep analysis of LLM benchmark quality.

![PSN-IRT](https://github.com/Joe-Hall-Lee/PSN-IRT/blob/main/assets/main-fig.png)

## Data
- `data/combine.csv`: This is the unified binary response matrix used for training the PSN-IRT model. It contains the evaluation results of 12 LLMs on 11 benchmarks, totaling 41,871 items. The results were obtained using [OpenCompass](https://github.com/open-compass/opencompass). Each entry (i,j) in the matrix is 1 if model i answered item j correctly, and 0 otherwise.
- `results/student_abilities.csv`: This file contains the estimated latent abilities ($\theta$) for each of the 12 models, as output by our trained PSN-IRT model. It reflects the psychometrically measured proficiency of each LLM across the evaluated benchmarks.
- `results/item_parameters.csv`: This file contains the estimated item parameters for all 41,871 benchmark items, as output by our trained PSN-IRT model. Each row corresponds to a single item and includes its estimated discriminability (a), difficulty (b), guessing-rate (c), and feasibility (d) parameters from the 4PL model.

## ⚡️ Usage

### Train PSN-IRT

To train the PSN-IRT model from scratch, you can run the train.py script. This will take the unified response matrix as input and produce the estimated model abilities and item parameters as output files.

```bash
python train.py \
    --max_epochs 30 \
    --batch_size 512 \
    --learning_rate 0.003
```

## Citation
```bibtex
@inproceedings{zhou2025lost,
      title={Lost in Benchmarks? Rethinking Large Language Model Benchmarking with Item Response Theory}, 
      author={Zhou, Hongli and Huang, Hui and Zhao, Ziqing and Han, Lvyuan and Wang, Huicheng and Chen, Kehai and Yang, Muyun and Bao, Wei and Dong, Jian and Xu, Bing and Zhu, Conghui and Cao, Hailong and Zhao, Tiejun},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      year={2026}
}
```