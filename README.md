# Scientific Computing Assignment 3

## Overview
Source Code for assignment 3.

## Repository Structure
```
SC_SET3/
├── fig/                           # Directory containing result images
├── src/                           # Source code directory
│   ├── eigenmode/                 # Part I: Eigenmodes of drums or membranes of different shapes
│   │   ├── gen_M.py               #  Question B: 4 approaches to generate the matrix M
│   │   ├── solve_eigenv.py        #  Question B: wrapper for different solvers
│   │   ├── speed_analysis.py      #  Question C: Solve the M
│   │   ├── spectrum.py            #  Question D: spectrum display
│   │   ├── u_solution.py          #  Question E: U animation over time
│   ├── create_grids.py            #  Question A
│   │ 
│   ├── steady_diffusion/          # Part II:  Direct methods for solving steady state problems
│   │ 
│   ├── optional/                  # Part III: The leapfrog method - efficient time integration
│   
├── .gitignore                     # Git ignore file
├── LICENSE                        # License information
├── presentation_3_1_and_3_2.ipynb # Jupyter Notebook for presentation
├── presentation_3_3.ipynb         # Jupyter Notebook for presentation
├── requirements.txt               # Dependencies list for Python installation
├── README.md                      # Project documentation
```

## Description of Files
### Source Code (`src/`)
- **matijs/**: Contains the discretized solution for Diffusion Limited Aggregation (DLA), including probabilistic growth modeling.
- **noa/**: Implements the Monte Carlo simulation for DLA using random walkers and includes comparison metrics.
- **alex/**: Includes the numerical solution for the Gray-Scott reaction-diffusion system, along with visualization scripts.

### Figures (`fig/`)
- This directory contains all result images generated from the computations.

### Other Files
- **LICENSE**: License file for the repository.
- **README.md**: This document, providing an overview of the project.

## Running the Code
To execute the different parts of the assignment, navigate to the **presentation_3_1_and_3_2.ipynb** and **presentation_3_3.ipynb**, which contains overview and visualizations.

## Dependencies
Ensure the following Python packages are installed:
```bash
pip install -r requirements.txt
```

## License
This project is licensed under the terms specified in the `LICENSE` file.

## Contact
For any questions, please refer to the presentation notebook or the provided scripts.
