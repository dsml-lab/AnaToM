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
@inproceedings{suzuki-etal-2025-anatom,
    title = "{A}na{T}o{M}: {A} {D}ataset {G}eneration {F}ramework for {E}valuating {T}heory of {M}ind {R}easoning {T}oward the {A}natomy of {D}ifficulty through {S}tructurally {C}ontrolled {S}tory {G}eneration",
    author = "Suzuki, Jundai  and
      Ishigaki, Ryoma  and
      Maeda, Eisaku",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-ijcnlp.14/",
    pages = "244--257",
    ISBN = "979-8-89176-303-6",
}
```

## License
This dataset is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/).<br>
This work is derived from and utilizes components of the [ToMi dataset](https://github.com/facebookresearch/ToMi/tree/master), which is also licensed under CC BY-NC 4.0.
