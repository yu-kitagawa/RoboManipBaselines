
---

# 🤖 [RoboManipBaselines](https://isri-aist.github.io/RoboManipBaselines-ProjectPage)

A software framework integrating various **imitation learning methods** and **benchmark environments** for robotic manipulation.  
Provides easy-to-use **baselines** for policy training, evaluation, and deployment.

[![CI-install](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/install.yml/badge.svg)](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/install.yml)
[![CI-pre-commit](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/pre-commit.yml)
[![LICENSE](https://img.shields.io/github/license/isri-aist/RoboManipBaselines)](https://github.com/isri-aist/RoboManipBaselines/blob/master/LICENSE)

https://github.com/user-attachments/assets/c37c9956-2d50-488d-83ae-9c11c3900992

https://github.com/user-attachments/assets/ba4a772f-0de5-47da-a4ec-bdcbf13d7d58

---

## 🚀 Quick Start

Start collecting data in the **MuJoCo** simulation, train your model, and rollout the ACT policy in just a few steps!  
📄 See the [Quick Start Guide](./doc/quick_start.md).

---

## ⚙️ Installation

Follow our step-by-step [Installation Guide](./doc/install.md) to get set up smoothly.

---

## 🧠 Policies

We provide several powerful policy architectures for manipulation tasks:

- 🔹 **[MLP](./robo_manip_baselines/policy/mlp)** – Simple feedforward policy  
- 🔹 **[SARNN](./robo_manip_baselines/policy/sarnn)** – Sequence-aware RNN-based policy  
- 🔹 **[ACT](./robo_manip_baselines/policy/act)** – Transformer-based imitation policy  
- 🔹 **[MT-ACT](./robo_manip_baselines/policy/mt_act)** – Multi-task Transformer-based imitation policy  
- 🔹 **[DiffusionPolicy](./robo_manip_baselines/policy/diffusion_policy)** – Diffusion-based behavior cloning policy  

---

## 📦 Data

- 📂 [Dataset List](./doc/dataset_list.md): Pre-collected expert demonstration datasets  
- 🧠 [Learned Parameters](./doc/learned_parameters.md): Trained model checkpoints and configs
- 📄 [Data Format](./doc/rmb_data_format.md): Description of the custom RMB data format used in RoboManipBaselines

---

## 🎮 Teleoperation

Use your own teleop interface to collect expert data.  
See [Teleop Tools](./robo_manip_baselines/teleop) for more info.

---

## 🌍 Environments

Explore diverse manipulation environments:

- 📚 [Environment Catalog](./doc/environment_catalog.md) – Overview of all task environments  
- 🔧 [Env Setup](./robo_manip_baselines/envs) – Installation guides per environment
- ✏️ [How to Add a New Environment](./doc/how_to_add_env.md) – Guide for adding a custom environment

---

## 🧰 Miscellaneous

Check out [Misc Scripts](./robo_manip_baselines/misc) for standalone tools and utilities.

---

## 📊 Evaluation Results

See benchmarked performance across environments and policies:  
📈 [Evaluation Results](./doc/evaluation_results.md)

---

## 🤝 Contributing

We welcome contributions!  
Check out the [Contribution Guide](./CONTRIBUTING.md) to get started.

---

## 📄 License

This repository is licensed under the **BSD 2-Clause License**, unless otherwise stated.  
Please check individual files or directories (especially `third_party` and `assets`) for specific license terms.

---

## 📖 Citation

If you use RoboManipBaselines in your work, please cite us:

```bibtex
@software{RoboManipBaselines_GitHub2024,
  author = {Murooka, Masaki and Motoda, Tomohiro and Nakajo, Ryoichi},
  title = {{RoboManipBaselines}},
  url = {https://github.com/isri-aist/RoboManipBaselines},
  version = {1.0.0},
  year = {2024},
  month = dec,
}
```

---
