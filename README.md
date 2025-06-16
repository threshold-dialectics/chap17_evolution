# Chapter 17 Experiment: Emergence of Network Form

This repository contains the Python simulation code for the experiment described in Chapter 17, "Threshold Dialectics and Emergence: Form and Networks by FEP-Driven Evolution," from the book *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness* by Axel Pond.

## Overview

This experiment explores a central thesis of Threshold Dialectics (TD): that the enduring structural "form" of a complex adaptive system is a **frozen history** of its long-term adaptation to environmental pressures. We hypothesize that persistent environmental conditions, characterized as distinct TD regimes, act as powerful selective forces that sculpt a system's architecture.

This simulation models the evolution of a resource distribution network among a population of agents. By subjecting the system to three different resource regimes—**Scarcity**, **Shock**, and **Flux**—we demonstrate how each environment gives rise to a distinct and FEP-optimal network topology.

> **Key Thesis:** Persistent environmental pressures, characterized as distinct TD dynamic regimes (e.g., chronic Scarcity, intermittent Shocks, high Flux), act as powerful selective forces. Guided by the Free Energy Principle (FEP), systems adapt their internal structure ("form") to be FEP-optimal for these chronic TD regimes.

## Core Concepts

*   **Threshold Dialectics (TD):** A framework for understanding how complex adaptive systems maintain viability by managing core capacities: perception gain ($g$), policy precision ($\beta$), and energetic slack ($F_{crit}$).
*   **TD Regimes:** The characteristic dynamic pressures an environment imposes on a system. This simulation models three such regimes based on resource availability.
*   **Free Energy Principle (FEP):** The fundamental imperative driving adaptive systems to minimize long-term surprise or prediction error. In this model, agents' heuristic rules for survival and connection are analogous to FEP-driven behaviors aimed at maintaining viability.
*   **Form:** The enduring structural and organizational patterns of a system. In this simulation, "form" is the emergent topology of the agent network.

## The Simulation Model

The experiment is an agent-based model (ABM) where "225" agents on a "15x15" grid evolve a resource-sharing network over "2000" time steps.

### The Agents

*   Each agent has an internal resource level ("fcrit"), a proxy for its energetic slack.
*   Agents must maintain "fcrit" above a survival threshold by acquiring resources while paying metabolic and link-maintenance costs.
*   Agents have an evolvable gene ("beta_lever") that influences their preferred number of network connections.
*   Agents form and break links with neighbors based on simple, FEP-aligned heuristics (e.g., seeking links when resources are sufficient, breaking links under stress).

### The Environment: TD Regimes

The simulation explores three distinct resource regimes:
1.  **Scarcity:** A constant, low-level inflow of resources is distributed evenly among all living agents. This environment selects for efficiency and conservation.
2.  **Shock:** A moderate baseline resource inflow is periodically interrupted by severe, system-wide resource droughts. This environment selects for robustness and resource-sharing capacity.
3.  **Flux:** A high total resource inflow is concentrated at a few randomly relocating "hotspots." This environment selects for agility, exploration, and adaptability.

### Evolutionary Dynamics

*   Agents whose "fcrit" drops below a survival threshold "die" and are removed from the network.
*   A dead agent's grid position is repopulated by an offspring of a randomly chosen successful neighbor.
*   The offspring inherits the parent's "beta_lever" gene with a small chance of mutation. This creates a selective pressure for strategies and network positions that enhance long-term viability.

## How to Run the Simulation

### Prerequisites

You will need Python 3.9+ and the following libraries:
*   "numpy"
*   "networkx"
*   "matplotlib"
*   "scipy"

You can install them from a "requirements.txt" file.

**requirements.txt:**
"""
numpy
networkx
matplotlib
scipy
"""

### Installation and Execution

1.  **Clone the repository:**
    """bash
    git clone https://github.com/your-username/threshold-dialectics-ch17.git
    cd threshold-dialectics-ch17
    """

2.  **Install the required packages:**
    """bash
    pip install -r requirements.txt
    """

3.  **Run the simulation:**
    The script "mvs_network_evolution.py" will run the entire experiment, simulating all three regimes with 30 replicates each.
    """bash
    python mvs_network_evolution.py
    """
    The simulation will print progress to the console and may take some time to complete. All outputs will be saved to a "results/" directory.

## Expected Output

The script will create a "results/" directory containing the following outputs for each of the three regimes ("scarcity", "shock", "flux"):

1.  **Final Network Snapshots:** A PNG image showing the final evolved network topology for the first replicate of each regime.
    *   "mvs_network_{regime}_run0_step_1999.png"

2.  **Averaged Metrics Evolution:** Two plots showing the mean and standard deviation of key network metrics over the 2000 simulation steps, averaged across all 30 replicates. Each plot contains four metrics for better readability.
    *   "mvs_metrics_{regime}_part1.png"
    *   "mvs_metrics_{regime}_part2.png"

3.  **Final Distributions:** Plots showing the aggregated distributions of node degree and component size at the end of all simulation runs for a regime.
    *   "mvs_degree_dist_{regime}.png"
    *   "mvs_component_size_dist_{regime}.png"

4.  **Simulation Summary:** A "summary.json" file containing a detailed quantitative summary of all parameters and final results, including mean and standard deviation for all recorded metrics.

## Results & Interpretation

The simulation robustly demonstrates that the three different TD regimes sculpt measurably distinct network forms, validating the core hypothesis.

### Evolved Network Topologies

Visual inspection of the final network structures reveals striking differences:

| Scarcity Regime | Shock Regime | Flux Regime |
| :---: | :---: | :---: |
| ![Scarcity Network](https://raw.githubusercontent.com/axpon/threshold-dialectics/main/Images/mvs_network_scarcity_run0_step_1999.png) | ![Shock Network](https://raw.githubusercontent.com/axpon/threshold-dialectics/main/Images/mvs_network_shock_run0_step_1999.png) | ![Flux Network](https://raw.githubusercontent.com/axpon/threshold-dialectics/main/Images/mvs_network_flux_run0_step_1999.png) |
| Sparse, fragmented, string-like structures that minimize link costs. | Dense, highly interconnected network optimized for resource sharing and shock absorption. | Agile, modular network of intermediate density, balancing reach with the cost of rewiring. |

### Quantitative Analysis

The final evolved network metrics, averaged over 30 replicates, show statistically significant differences across the regimes, confirming the qualitative visual findings.

| Metric | Scarcity Regime | Shock Regime | Flux Regime |
| :--- | :--- | :--- | :--- |
| **Avg. Degree** | $1.99 \pm 0.12$ | **$3.07 \pm 0.10$** | $2.50 \pm 0.15$ |
| **Avg. Clustering** | $0.077 \pm 0.027$ | **$0.179 \pm 0.024$** | $0.136 \pm 0.022$ |
| **LCC Avg. Path Length**| $11.69 \pm 2.74$ | **$9.84 \pm 0.57$** | $11.56 \pm 1.86$ |
| **Box Dimension (2D)** | $1.65 \pm 0.15$ | **$1.91 \pm 0.01$** | $1.83 \pm 0.08$ |

*   The **Shock** regime evolves the densest, most clustered, and most space-filling networks, an FEP-optimal form for distributing resources and withstanding systemic shocks. This form enables a high reliance on slack ($w_3$).
*   The **Scarcity** regime evolves the sparsest networks, an FEP-optimal form for minimizing the metabolic cost of maintaining links. This form enables a high reliance on precision/efficiency ($w_2$).
*   The **Flux** regime evolves networks of intermediate density, a trade-off between connectivity (to find new resources) and agility (to avoid the cost of a fully connected structure). This form enables a high reliance on perception/adaptability ($w_1$).

These results provide strong evidence that form is not arbitrary but is an emergent, FEP-optimal adaptation to the long-term statistical structure of the environment, a frozen echo of its TD history.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.