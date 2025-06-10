# Structuring the Processing Frameworks for Data Stream Evaluation and Application

The definition of structured frameworks stems from the need to reliably assess data stream classification methods, considering the constraints of delayed and limited label access. This repository contains four processing schemes that link the tasks of drift detection and classification while considering a natural phenomenon of label delay. 
The presented research shows that classification quality is significantly affected not only by the phenomenon of concept drift and label delay, but also by the undertaken processing scheme that defines the flow of labels in the recognition system. Considering a specific processing framework depending on real-world constraints proves to be a critical aspect of reliable and realistic experimental evaluation.

## Contents:

### *detectors* -- implementation of drift detectors
### *fig_frameworks* -- figures presenting results of experiments
### *frameworks* -- implementation of frameworks
### *results* -- results of experiments
### *tables* -- plots of tables and tables

#
### *experiment.py* -- experiment on synthetic data streams
### *experiment_covtype.py* -- experiment on covtype data streams

#
### *experiment_vis.py* -- visualizing the results of synthetic data experiment 
### *experiment_covtype_vis.py* -- visualizing the results of covtype experiment

### *experiment_table.py* -- script for generating and plotting tables for synthetic data
### *experiment_covtype_table.py* -- generating tables for results of covtype experiment
### *experiment_covtype_stat.py* -- critical difference diagrams and statistical analysis of covtype results
