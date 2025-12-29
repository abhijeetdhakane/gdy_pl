# GDyNet-Ferro Documentation

Welcome to the GDyNet-Ferro documentation.

## Overview

**GDyNet-Ferro** is a graph neural network framework for identifying slow dynamical features and hidden states in molecular dynamics simulations. This PyTorch implementation uses the **Variational Approach for Markov Processes (VAMP)** to learn meaningful collective variables from atomistic trajectories, with a focus on ferroelectric materials.

![Overview Architecture](asset/fig1_revised_hdr2.png)



## Quick Links

- [README](readme.md) - Main documentation
- [Training Guide](training-guide.md) - Comprehensive training instructions
- [Paper](https://doi.org/10.1016/j.cartre.2023.100264) - Carbon Trends 2023 publication

## Installation

```bash
git clone https://github.com/abhijeetdhakane/gdy_pl.git
cd gdy_pl
pip install -e .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{dhakane2023graph,
  title={A Graph Dynamical Neural Network Approach for Decoding Dynamical States in Ferroelectrics},
  author={Dhakane, Abhijeet and Xie, Tian and Yilmaz, Dundar and van Duin, Adri and Sumpter, Bobby G and Ganesh, P},
  journal={Carbon Trends},
  volume={11},
  pages={100264},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.cartre.2023.100264}
}
```


[image1]: asset/fig1_revised_hdr2.png
