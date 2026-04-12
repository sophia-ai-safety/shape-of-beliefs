## Overview

Code and artifacts for the Goodfire research paper *The Shape of Beliefs: Geometry, Dynamics, and Interventions along Representation Manifolds of Language Models' Posteriors*. Preprint: [arXiv:2602.02315](https://arxiv.org/abs/2602.02315)

## Setup

1. Clone the repo:
   ```bash
   git clone https://github.com/raphael-goodfire/shape-of-beliefs.git
   cd shape-of-beliefs
   ```

2. Create the virtual environment and install dependencies with `uv`:
   ```bash
   uv venv
   uv pip install -e .
   ```

## Workflow

1. Generate data (sequences, activations, logits; about 50GB):
   ```bash
   chmod +x scripts/generate_all.sh
   ./scripts/generate_all.sh
   ```

2. Generate figures from the notebooks in `figures/`.

3. Visualize different steering schemes interactively from the streamlit app:
   ```bash
   uv run streamlit run app/steering_explorer_app.py
   ```

### Citation

```bibtex
@article{Sarfati2026ShapeOfBeliefs,
  title={The Shape of Beliefs: Geometry, Dynamics, and Interventions along Representation Manifolds of Language Models' Posteriors},
  author={Sarfati, Rapha{\"e}l and Bigelow, Eric and Wurgaft, Daniel and Merullo, Jack and Geiger, Atticus and Lewis, Owen and McGrath, Tom and Singh Lubana, Ekdeep},
  journal={arXiv preprint arXiv:2602.02315},
  year={2026},
  month={feb},
  eprint={2602.02315},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2602.02315},
  url={https://arxiv.org/abs/2602.02315}
}
```

### License
MIT

### Contact
raphael@goodfire.ai
