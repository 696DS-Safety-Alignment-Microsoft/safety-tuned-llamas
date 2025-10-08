# How to Fine-Tune Safely on a Budget: Model Adaptation Using Minimal Resources

**Abstract**
Supervised fine-tuning (SFT) on benign data can paradoxically erode a language model's safety alignment, a phenomenon known as catastrophic forgetting of safety behaviors. Although prior work shows that randomly adding safety examples can reduce harmful output, the principles that make certain examples more effective than others remain poorly understood. This paper investigates the hypothesis that the effectiveness of a safety example is governed by two key factors: its instruction-response behavior (e.g., refusal vs. explanation) and its semantic diversity across harm categories. We systematically evaluate sampling strategies based on these axes and find that structured, diversity-aware sampling significantly improves model safety. Our method reduces harmfulness by up to 41\% while adding only 0.05\% more data to the fine-tuning set. 


## Starting Point

This repo contains code that was modified from the below paper for our specific research purpose [1][2]

SafetyDatasets are available under the `data/evaluation` directory.

Training data is available under the `data/training` directory. Where you will find the instruction-output pairs.

Selection data is available under the `data/selection` directory. Where you will find instruction-output pairs for the SafetyDataset along with a score that facilitates intelligent subsampling

## Tuning and Generation

Fine-tuning code and generation come from [Alpaca-LoRa](https://github.com/tloen/alpaca-lora) repository.

Wildguard classification and sampling code for PSS and SS can be found in the `sample` directory. After sampling the safety points, please use `data/data_creation.py` to generate the final fine-tuning dataset.


## Script to run 
- `scripts/run-job-training-{LLM}.sbatch` fine-tunes the LLM with provided data-path
- `scripts/run-job-generation-{LLM}.sbatch` generates responses to an evaluation dataset consisting of instructions from BeaverTails-Evaluation (to test the fine-tuned LLM's safety)
- `scripts/run-job-generation-{LLM}-local.sbatch` generates responses to an evaluation dataset consisting of instructions from I-Alpaca (to test the fine-tuned LLM's helpfulness)
- `scripts/run-job-evaluation-{LLM}.sbatch` runs the Harm Reward Model to evaluate harmfulness in generated responses
- `scripts/run-job-evaluation-{LLM}-helpfulness.sbatch` runs the OpenAssistant-DoBERTA Reward Model to evaluate helpfulness in generated responses

## Visualizations
- `notebooks/score_subsamples.ipynb` to generate a line plot of dataset scores across random trials and optimized trials
- `notebooks/visualize_results.ipynb` to generate a comparative study of LLM baseline performance and optimized trials

# Licensing

* Code is licensed under the MIT License. 

* Due to the fact that some of the data is GPT-generated and comes from other work, Data is licensed under the Creative Commons Attribution Non Commercial 4.0 License. For SafeText data, also referred as PhysicalSafety in our paper, please refer to [1].

[1] Levy, S., Allaway, E., Subbiah, M., Chilton, L., Patton, D., McKeown, K., & Wang, W. Y. (2022). Safetext: A benchmark for exploring physical safety in language models. EMNLP.

[2] Bianchi, F., Suzgun, M., Attanasio, G., Röttger, P., Jurafsky, D., Hashimoto, T., & Zou, J. (2024). Safety-Tuned LLaMAs: Lessons from improving the safety of large language models that follow instructions. In Proceedings of the Twelfth International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=gT5hALch9z

## Citation

If you use our dataset, code, or results, please cite:

```bibtex
@inproceedings{pham-etal-2025-finetune-safely-budget,
  title     = {How to Fine-Tune Safely on a Budget: Model Adaptation Using Minimal Resources},
  author    = {Pham, Anh C. and Thalanki, Mihir and Sun, Michael and Chaloo, Aditya and Gupta, Ankita and Xia, Tian and Mate, Aditya and Nosakhare, Ehi and Srinivasan, Soundararajan},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  year      = {2025},
  publisher = {Association for Computational Linguistics},
  url       = {https://openreview.net/forum?id=fPWXX4vsrp},
  note      = {Industry Track, EMNLP 2025. Camera-ready version.}
}
