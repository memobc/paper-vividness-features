# The Paper
This repository includes data and code for the following paper:

**Cooper, R.A. & Ritchey, M. (under review). Patterns of episodic content and specificity predicting subjective memory vividness.**

# Abstract

The ability to remember and internally represent events is often accompanied by a subjective sense of “vividness”. Vividness measures are frequently used to evaluate the experience of remembering and imagining events, yet little research has considered the objective attributes of event memories that underlie this subjective judgment, and individual differences in this mapping. Here, we tested how the content and specificity of event memories support subjectively vivid recollection. Over three experiments, participants encoded events containing a theme word and three distinct elements — a person, a place, and an object. In a memory test, memory for event elements was assessed at two levels of specificity — semantic gist (names) and perceptual details (lure discrimination). We found a strong correspondence between memory vividness and memory for gist information that did not vary by which elements were contained in memory. There was a smaller, additive benefit of remembering specific perceptual details on vividness, which, in one study, was driven by memory for place details. Moreover, we found individual differences in the relationship between memory vividness and objective memory attributes primarily along the specificity dimension, such that one cluster of participants used perceptual detail to inform memory vividness whereas another cluster were more driven by gist information. Therefore, while gist memory appears to drive vividness on average, there were idiosyncrasies in this pattern across participants. When assessing subjective ratings of memory and imagination, research should consider how these ratings map onto objective memory attributes in the context of their study design and population.

# Data

[`data`](https://github.com/memobc/paper-vividness-features/tree/main/data) per participant were saved as a csv file with trial information and corresponding responses per row.

# Code

Jupyter notebooks in the [`code`](https://github.com/memobc/paper-vividness-features/tree/main/code) folder contain all analyses for the within-experiment analyses and the individual differences analyses. Separate `.py` files provide custom functions to the notebooks. Data are loaded in from the `data` folder.
* `vividness_analysis_Exp[1/2/3].ipynb`: runs analyses within each experiment, first quality-checking the data, and then analyzing the relationship between event attributes and memory vividness. Each notebook also provides an overview of the task design.
* `individual_differences_analysis.ipynb`: runs analyses across all participants, testing how mean memory performance is related to subjective evaluations of memory, and if the *pattern* of relationships between objective and subjective memory measures differs across individuals.

# License
All code is licensed under the [MIT license](https://github.com/memobc/paper-vividness-features/blob/main/LICENSE).

# Comments?
Please direct any comments or questions to Rose Cooper, r.cooper at northeastern.edu. Please feel free to use any of these scripts. Unfortunately I cannot provide support for you to adapt them to your own data. Notice a bug? Please tell me!
