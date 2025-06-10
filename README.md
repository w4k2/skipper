# Structuring the Processing Frameworks for Data Stream Evaluation and Application

The definition of structured frameworks stems from the need to reliably assess data stream classification methods, considering the constraints of delayed and limited label access. This repository contains four processing schemes that link the tasks of drift detection and classification while considering the natural phenomenon of label delay.

The presented research shows that classification quality is significantly affected not only by the phenomena of concept drift and label delay, but also by the adopted processing scheme, which defines the flow of labels in the recognition system. Considering a specific processing framework depending on real-world constraints proves to be a critical aspect of reliable and realistic experimental evaluation.

## Contents

### Folders

- **detectors** — implementation of drift detectors  
- **fig_frameworks** — figures presenting results of experiments  
- **frameworks** — implementation of frameworks  
- **results** — results of experiments  
- **tables** — plots of tables and tabular results  

### Scripts

- **exp_syn.py** — experiment on synthetic data streams  
- **exp_covtype.py** — experiment on Covtype data streams  

- **syn_vis.py** — visualizing the results of the synthetic data experiment  
- **syn_vis_single.py** — visualizing the results of the synthetic data experiment for a single data stream flow  
- **covtype_vis.py** — visualizing the results of the Covtype experiment  

- **syn_table.py** — script for generating and plotting tables for synthetic data  
- **covtype_table.py** — generating tables for results of the Covtype experiment  
- **covtype_stat.py** — critical difference diagrams and statistical analysis of Covtype results  

- **exp_pre.py** — preliminary experiment on virtual and real drifts  
