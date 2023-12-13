# threestudio-shap-E
Shap-E guidance extension of threestudio. The original implementation can be found in [https://github.com/openai/shap-e](https://github.com/openai/shap-e). Currently, it is only used for initialization of Gaussian Splatting.

## Installation
```
cd custom
git clone https://github.com/DSaurus/threestudio-shap-e
git clone https://github.com/openai/shap-e.git
pip install -e shap-e
```

## Examples
Please see [threestudio-3dgs](https://github.com/DSaurus/threestudio-3dgs#load-from-ply) for more details.

## Citation
```
@article{jun2023shap,
  title={Shap-e: Generating conditional 3d implicit functions},
  author={Jun, Heewoo and Nichol, Alex},
  journal={arXiv preprint arXiv:2305.02463},
  year={2023}
}
```
