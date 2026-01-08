# dimerization_network_expressivity

Code from [Parres-Gold et al. *Cell* **2025**](https://doi.org/10.1016/j.cell.2025.01.036) to analyze the computational capabilities of competitive protein dimerization networks.

All code, raw data, and processed data is available on [CaltechDATA](https://doi.org/10.22002/1gffr-va537).

## Utilities
`dimer_network_utilities.py` provides functions that are used by other notebooks in the repository.

## Resources
The following resources are available to simulate the input-output functions of arbitrary networks:

|  |  |
| ---|---|
| `simulate_individual_networks.ipynb` | This notebook demonstrates simulations of individual one- and two-input networks with schematics of the network architecture.|
| [Interactive Google Colab](https://bxky.short.gy/interactive_dimerization_networks) | Interactive Google Colab notebook for simulating one- and two-input networks.|

## Notebooks to Replot Figures
The following notebooks were used to plot the figures used in the paper. They were created to make re-plotting of the data as simple as possible. The data for these figures are drawn directly from the archived data folder. These notebooks also include many plots not included in the main or supplementary figures; such plots have not been aesthetically cleaned up. 

|  |  |
| ---|---|
| `remake_GraphicalAbstract.ipynb` | Plots for graphical abstract |
| `remake_Fig1.ipynb` |	Plots for Figure 1 |
| `remake_Fig2.ipynb` |	Plots for Figure 2 |
| `remake_Fig3.ipynb` |	Plots for Figure 3 |
| `remake_Fig4.ipynb` |	Plots for Figure 4 and Figure S3 |
| `remake_Fig5.ipynb` |	Plots for Figure 5 and Figure S4 |
| `remake_Fig6.ipynb` |	Plots for Figure 6 and Figure S6 |
| `remake_FigS5.ipynb` |	Plots for Figure S5 |
| `remake_FigS7.ipynb` |	Plots for Figure S7 |


## Notebooks to Demonstrate Analysis Methods
The following notebooks were created from code that was used to perform the majority of the analysis for the paper. Many of these notebooks use the ray package for parallelization, which is not available for Windows. Thus, if you would like to re-run these notebooks, make sure to disable the commands involving ray. It may also be necessary to change code relevant to loading and saving files.

|  |  |
| ---|---|
| `generate_param_screen.ipynb` |	This notebook performs a parameter screen of networks, considering 1 or 2 monomers as the input. |
| `analyze_1D_screen.ipynb` |	This notebook analyzes the responses observed in the parameter screen of one-input network responses. |
| `analyze_2D_screen.ipynb` |	This notebook analyzes the responses observed in the parameter screen of two-input network responses. |
| `analyze_connectivity_trends.ipynb` |	Using the large screen of many networks, this notebook assesses how various network properties vary with network connectivity. |
| `optimizer_dualannealing.ipynb` |	This notebook demonstrates how a dual annealing algorithm was used to optimize dimerization networks to perform desired functions. |
| `boolean_complexity.ipynb` |	This notebook demonstrates how to calculate the Boolean complexity of the Boolean functions used in this work. |
| `analyze_equilibration_kinetics.ipynb` |	This notebook uses deterministic simulations, via numerical integration of ordinary differential equations (ODEs), to estimate the time required for network re-equilibration after a perturbation of the input monomer. |
| `analyze_intrinsic_noise.ipynb` |	This notebook will use stochastic stimulations, via Gillespie simulation, of networks at steady state to estimate stochastic fluctuations in dimer concentrations at equilibrium. |
| `analyze_expression_noise_robustness.ipynb` |	This notebook simulates various networks (one for each unique function) under random perturbations of monomer expression levels (total abundances). |



 



