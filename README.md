# 2024 U.S. Election Twitter Network Analysis

Implementation of a large-scale network analysis pipeline to study information propagation dynamics.

## Implementation

* **build_network.py**: Data pipeline that transformed raw Twitter metadata into a directed graph of 589,895 nodes and 13,671 edges.
* **1.ipynb**: 
    * **Topological Analysis**: Verified scale-free properties and hierarchical "megaphone" topology.
    * **Epidemic Modeling**: Custom implementation of the Susceptible-Infected (SI) model for diffusion simulation.
    * **Evaluation**: Applied grid search for $\beta$ parameter estimation and MAPE for model-to-reality fit assessment.

## Key Results

* **High Modularity**: Detected 60 communities (modularity 0.895), revealing significant discourse fragmentation.
* **Mixed Diffusion**: 63.6% of cascades are driven by a combination of network structure and external factors.
