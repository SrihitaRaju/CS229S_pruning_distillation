# Code Model Distillation Project

## Purpose

This project aims to determine the best methods for creating a distilled model for coding tasks from larger open-source models. We will implement and analyze various compression techniques, including pruning and distillation, to assess their impact on end benchmarks.

## Project Structure

- `src/`: Source code for the project
  - `data_loader.py`: Script to download and prepare the Llama 3.1 8B model and coding dataset
  - `distillation/`: Distillation techniques implementation
  - `pruning/`: Pruning techniques implementation
  - `evaluation/`: Benchmark and evaluation scripts
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `config/`: Configuration files
- `results/`: Directory to store experiment results

## Getting Started

1. Clone this repository
2. Install the required dependencies (requirements.txt will be provided)
3. Run `src/data_loader.py` to download the necessary models and datasets
4. Explore the various compression techniques in the `src/distillation/` and `src/pruning/` directories
5. Use the evaluation scripts in `src/evaluation/` to benchmark the compressed models

## Model Information

We are using the SalesForce 2B model

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

Note: The use of the Llama 3.1 8B model is subject to the [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3%5F1/LICENSE). Make sure to review and comply with its terms.
