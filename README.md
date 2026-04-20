# ABC: Any-Subset Autoregression via Non-Markov Diffusion Bridges in Continuous Space and Time

[Gabe Guo](https://gabeguo.github.io), [Thanawat Sornwanee](https://tsornwanee.github.io/), [Lutong Hao](https://www.linkedin.com/in/lutong-hao/), [Elon Litman](https://elonlit.com/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), [Jose Blanchet](https://joseblanchet.com/).

_Generative modeling continuous-time, continuous-space stochastic processes (e.g., videos, weather forecasts) conditioned on partial observations (e.g., first and last frames) is a fundamental challenge.
Existing approaches, (e.g., diffusion models), suffer from key limitations: (1) noise-to-data evolution fails to capture structural similarity between states close in physical time and has unstable integration in low-step regimes; (2) random noise injected is insensitive to the physical process's time elapsed, resulting in incorrect dynamics; (3) they overlook conditioning on arbitrary subsets of states (e.g., irregularly sampled timesteps, future observations).
We propose ***ABC***: ***A***ny-Subset Autoregressive Models via Non-Markovian Diffusion ***B***ridges in ***C***ontinuous Time and Space. Crucially, we model the process with ***one continual SDE*** whose time variable and intermediate states track the ***real time and process states***. 
This has provable advantages: (1) the starting point for generating future states is the already-close previous state, rather than uninformative noise; (2) random noise injection scales with physical time elapsed, encouraging physically plausible dynamics with similar time-adjacent states.
We derive SDE dynamics via changes-of-measure on path space, yielding another advantage: (3) path-dependent conditioning on arbitrary subsets of the state history and/or future.
To learn these dynamics, we derive a path- and time-dependent extension of denoising score matching.
Our experiments show ***ABC***'s superiority to competing methods on multiple domains, including video generation and weather forecasting._

## Project Page and Visualizations

See visualizations [here](https://abc-diffusion.github.io/).

## Setting Up Environment

Python 3.12.12
```
pip install -r requirements.txt
```

We run experiments on a node of 4 A100s (80GB each). Depending on your cluster, you may need to change `--extra-index-url` on the first line to match your CUDA version.

## Obtaining the Data

[CelebV-HQ](https://celebv-hq.github.io/) latents can be found on [HuggingFace](https://huggingface.co/datasets/therealgabeguo/abc_data/tree/main/celebv_hq). [Sky-Timelapse](https://github.com/zhangzjn/DTVNet) latents can be found on [HuggingFace](https://huggingface.co/datasets/therealgabeguo/abc_data/tree/main/sky_timelapse/res-256x256-fpc-64). The script should auto-download them, as long as you have the correct format for the filepath (has to match the HF repo structure; default arguments already satisfy this).

Alternatively, if you have your own custom dataset, feel free to try out `encode_latents.py`.

## Obtaining the Checkpoints

All the checkpoints evaluated in the paper are available on [HuggingFace](https://huggingface.co/therealgabeguo/abc/tree/main).

## Training the Model

We have _TODO_ for where you need to fill in the path. You can also play around with hyperparameters.

Regular node:
```
bash _run_train.sh
```

Slurm:
```
sbatch _run_train.sh
```

## Running Inference

This auto-downloads the pretrained models we have from HF.

Change the paths and look at the _TODO_ in the script before running. You can change the flags as you see fit.

Regular node:
```
bash _run_inference.sh
```

Slurm:
```
sbatch _run_inference.sh
```

## Citation
```
@article{guo2026_abc,
  title={ABC: Any-Subset Autoregression via Non-Markovian Diffusion Bridges in Continuous Time and Space},
  author={Guo, Gabe and Sornwanee, Thanawat and Hao, Lutong and Litman, Elon and Ermon, Stefano and Blanchet, Jose},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgements

This repository builds off [DiT](https://github.com/facebookresearch/DiT/tree/main).