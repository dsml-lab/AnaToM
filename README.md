# AnaToM

## Project Structure

This repository is organized into two main directories: `dataset` for generation/analysis and `evaluate_model` for benchmarking.

```text
.
├── dataset/                             # Data generation & analysis module
│   ├── generate_benchmark_story_detect.py # [Main] Script to generate stories based on patterns
│   ├── create_test.py                   # Utilities to format or split test data
│   ├── analyze_patterns.py              # Script to analyze dataset distribution/statistics
│   ├── analyze_patterns_accuracy.py     # Script to validate pattern consistency
│   ├── world.json                       # [Config] Definitions of objects, locations, and agents
│   ├── pattern.json                     # [Config] Templates for Theory of Mind patterns
│   ├── stories.json                     # [Output] Generated story text
│   ├── qa_sets.json                     # [Output] Generated Question-Answer pairs
│   └── distribution_analysis.json       # [Output] Analysis report of the dataset
│
├── evaluate_model/                      # LLM evaluation module
│   ├── evaluate_gpt.py                  # Evaluation script for OpenAI GPT models
│   └── evaluate_llama.py                # Evaluation script for Llama models
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt                     # Python dependencies
```

## Citation
If you find our framework useful, please cite our work.
```
@inproceedings{suzuki2025AnaToM,
    title = "AnaToM: A Dataset Generation Framework for Evaluating Theory of Mind Reasoning toward the Anatomy of Difficulty through Structurally Controlled Story Generation",
    author = "Suzuki, Jundai  and
      Ishigaki, Ryoma  and
      Maeda, Eisaku",
    booktitle = "Findings of the Association for Computational Linguistics: IJCNLP-AACL 2025 (Findings)",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "Association for Computational Linguistics",
}
```

## License
This dataset is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/).<br>
This work is derived from and utilizes components of the [ToMi dataset](https://github.com/facebookresearch/ToMi/tree/master), which is also licensed under CC BY-NC 4.0.
