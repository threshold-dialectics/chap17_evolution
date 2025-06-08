import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import collections
import os  # For directory creation and path joining
import json

# --- Global Configuration ---
RESULTS_DIR = "results" # Define the output directory name
SUMMARY_FILE = os.path.join(RESULTS_DIR, "summary.json")
PREAMBLE_TEXT = """
objective: >
  Explore how chronic resource regimes influence the evolution of agent networks.\
  Different inflow patterns are expected to produce distinct topologies.
replications: 3
random_seeds: [42, 43, 44]
method_notes: >
  `simple_box_dim_2d` estimates a 2D box-counting dimension by counting occupied\
  grid boxes across logarithmically spaced sizes and fitting a line to the\
  resulting log-log curve.
definitions:
  avg_fcrit: "average critical resource threshold per agent (resource units)"
  n_alive: "number of living agents"
  avg_degree: "mean node degree among living agents"
  density: "graph edge density (0-1)"
  avg_clustering: "average clustering coefficient"
  overall_avg_path_length: "average shortest path length across the living graph; NaN if disconnected"
  n_components: "number of connected components"
  largest_comp_size: "size of the largest component"
  avg_preferred_degree: "mean preferred degree across agents"
  avg_beta_lever: "mean beta_lever value (unitless)"
  lcc_avg_degree: "average degree within the largest component"
  lcc_avg_path_length: "average path length within the largest component; NaN if <2 nodes"
  simple_box_dim_2d: "2D box-counting dimension estimate of the largest component"
units:
  inflow: "resource units per step"
  fcrit: "resource units"
notes: >
  'overall_avg_path_length' is recorded as NaN when the graph is disconnected.
"""

EPSILON = 1e-6
# --- Simulation Parameters ---
GRID_SIZE = 15
N_AGENTS = GRID_SIZE * GRID_SIZE
SIMULATION_STEPS = 2000
REPORT_INTERVAL = 200
# Down-sample the stored time series aggressively to keep the JSON output small
# (roughly every 50 steps with the final step always included)
DOWNSAMPLE_INTERVAL = 50

# Agent Parameters
INITIAL_FCRIT = 10.0
MIN_FCRIT_SURVIVAL = 1.0
METABOLISM_COST = 0.2
LINK_MAINTENANCE_COST = 0.05
PREFERRED_DEGREE_MIN = 1
PREFERRED_DEGREE_MAX = 4
MUTATION_RATE_STRATEGY = 0.1
MUTATION_STRENGTH_STRATEGY = 1
FCRIT_COLOR_VMAX = INITIAL_FCRIT * 1.5 # Standardized color scale

# TD related parameters
BETA_TO_DEGREE_FACTOR = 2.0
MUTATION_STRENGTH_BETA = 0.1

# Linking Heuristics
LINK_FORMATION_PROB_BASE = 0.3
LINK_BREAK_PROB_CRITICAL = 0.5
FCRIT_COMFORT_THRESHOLD_FACTOR = 1.5
FCRIT_CRITICAL_THRESHOLD_FACTOR = 0.5

# Resource Regimes Parameters
SCARCITY_INFLOW = 0.3 * N_AGENTS
SHOCK_BASE_INFLOW = 0.8 * N_AGENTS
SHOCK_REDUCED_INFLOW = 0.05 * N_AGENTS
SHOCK_DURATION = 50
SHOCK_INTERVAL = 250
FLUX_BASE_INFLOW = 0.8 * N_AGENTS
FLUX_N_SOURCES = 5
FLUX_SOURCE_LIFESPAN = 75

# Simulation Control
N_ITERATIONS_PER_REGIME = 30
BASE_SEED = 42


# --- Agent Class (No changes needed here) ---
class Agent:
    def __init__(self, agent_id, pos, initial_fcrit, beta_lever=1.0):
        self.id = agent_id
        self.pos = pos
        self.fcrit = initial_fcrit
        self.beta_lever = beta_lever
        self.preferred_degree = PREFERRED_DEGREE_MIN
        self.is_alive = True
        self._update_preferred_degree()

    def _update_preferred_degree(self):
        self.preferred_degree = int(
            np.clip(
                self.beta_lever * BETA_TO_DEGREE_FACTOR,
                PREFERRED_DEGREE_MIN,
                PREFERRED_DEGREE_MAX,
            )
        )

    def update_fcrit(self, inflow):
        if not self.is_alive:
            return
        self.fcrit += inflow
        self.fcrit -= METABOLISM_COST

    def should_form_link(self, current_degree, potential_neighbor_agent, graph):
        if not self.is_alive or not potential_neighbor_agent.is_alive:
            return False
        if self.id == potential_neighbor_agent.id:
            return False
        if graph.has_edge(self.id, potential_neighbor_agent.id):
            return False

        comfort_threshold = (METABOLISM_COST + self.preferred_degree * LINK_MAINTENANCE_COST) * FCRIT_COMFORT_THRESHOLD_FACTOR
        prob_factor = 1.0
        if self.fcrit < comfort_threshold:
            prob_factor = 1.5
        if current_degree < self.preferred_degree:
            prob_factor *= 1.5
        return random.random() < (LINK_FORMATION_PROB_BASE * prob_factor)


    def should_break_link(self, current_degree, link_partner_agent):
        if not self.is_alive:
            return False
        critical_threshold = MIN_FCRIT_SURVIVAL + FCRIT_CRITICAL_THRESHOLD_FACTOR
        if self.fcrit < critical_threshold and random.random() < LINK_BREAK_PROB_CRITICAL:
            return True
        if current_degree > self.preferred_degree and random.random() < 0.1:
            return True
        return False

    def die(self):
        self.is_alive = False
        self.fcrit = 0

    def __repr__(self):
        return (
            f"A{self.id}(fc:{self.fcrit:.1f}, pd:{self.preferred_degree}, "
            f"beta:{self.beta_lever:.2f}, alive:{self.is_alive})"
        )

# --- Environment Class (plot_network method is fine, other methods unchanged) ---
class MVSEnvironment:
    def __init__(self, size, n_agents, regime_type="scarcity"):
        self.size = size
        self.n_agents = n_agents
        self.agents = {}
        self.agent_grid = np.full((size, size), None, dtype=object)
        self.graph = nx.Graph()
        self.current_step = 0
        self.regime_type = regime_type
        self.flux_sources = []
        self.steps_since_last_flux_change = 0
        self.steps_since_last_shock = 0
        self.in_shock_period = False
        self.shock_steps_remaining = 0
        self._initialize_agents()
        if self.regime_type == "flux":
            self._update_flux_sources()

    def _initialize_agents(self):
        agent_id_counter = 0
        for r in range(self.size):
            for c in range(self.size):
                if agent_id_counter < self.n_agents:
                    beta = random.uniform(0.8, 1.2)
                    agent = Agent(agent_id_counter, (r, c), INITIAL_FCRIT, beta_lever=beta)
                    self.agents[agent_id_counter] = agent
                    self.agent_grid[r, c] = agent
                    self.graph.add_node(agent_id_counter, agent_obj=agent)
                    agent_id_counter += 1

    def get_neighbors(self, agent_pos):
        neighbors = []
        r, c = agent_pos
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    if self.agent_grid[nr, nc] is not None and self.agent_grid[nr,nc].is_alive:
                        neighbors.append(self.agent_grid[nr, nc])
        return neighbors

    def _distribute_resources(self):
        total_inflow_this_step = 0
        if self.regime_type == "scarcity":
            total_inflow_this_step = SCARCITY_INFLOW
        elif self.regime_type == "shock":
            if self.in_shock_period:
                total_inflow_this_step = SHOCK_REDUCED_INFLOW
                self.shock_steps_remaining -= 1
                if self.shock_steps_remaining <= 0:
                    self.in_shock_period = False
            else:
                total_inflow_this_step = SHOCK_BASE_INFLOW
                self.steps_since_last_shock += 1
                if self.steps_since_last_shock >= SHOCK_INTERVAL:
                    self.in_shock_period = True
                    self.shock_steps_remaining = SHOCK_DURATION
                    self.steps_since_last_shock = 0
        elif self.regime_type == "flux":
            total_inflow_this_step = FLUX_BASE_INFLOW
            self.steps_since_last_flux_change +=1
            if self.steps_since_last_flux_change >= FLUX_SOURCE_LIFESPAN:
                self._update_flux_sources()
                self.steps_since_last_flux_change = 0

        living_agents = [a for a in self.agents.values() if a.is_alive]
        if not living_agents:
            return

        if self.regime_type == "flux":
            if not self.flux_sources:
                 self._update_flux_sources()
            inflow_per_source_agent = total_inflow_this_step / len(self.flux_sources) if self.flux_sources else 0
            for r_s, c_s in self.flux_sources:
                agent_at_source = self.agent_grid[r_s,c_s]
                if agent_at_source and agent_at_source.is_alive:
                    agent_at_source.update_fcrit(inflow_per_source_agent)
        else:
            inflow_per_agent = total_inflow_this_step / len(living_agents)
            for agent in living_agents:
                agent.update_fcrit(inflow_per_agent)


    def _update_flux_sources(self):
        self.flux_sources = []
        for _ in range(FLUX_N_SOURCES):
            r, c = random.randint(0, self.size-1), random.randint(0, self.size-1)
            self.flux_sources.append((r,c))

    def _apply_link_maintenance_costs(self):
        for agent_id, agent in self.agents.items():
            if agent.is_alive:
                num_links = self.graph.degree(agent_id)
                agent.fcrit -= num_links * LINK_MAINTENANCE_COST

    def _update_network(self):
        agent_ids = list(self.agents.keys())
        random.shuffle(agent_ids)

        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            if not agent.is_alive:
                continue

            current_degree = self.graph.degree(agent_id)
            grid_neighbors = self.get_neighbors(agent.pos)

            if current_degree < 8:
                potential_partners = [n_agent for n_agent in grid_neighbors if not self.graph.has_edge(agent_id, n_agent.id)]
                if potential_partners:
                    chosen_partner = random.choice(potential_partners)
                    if agent.should_form_link(current_degree, chosen_partner, self.graph):
                        self.graph.add_edge(agent_id, chosen_partner.id)
                        current_degree +=1

            if current_degree > 0:
                current_links = list(self.graph.neighbors(agent_id))
                random.shuffle(current_links)
                for partner_id in current_links:
                    if not self.graph.has_edge(agent_id, partner_id): continue
                    partner_agent = self.agents[partner_id]
                    if agent.should_break_link(self.graph.degree(agent_id), partner_agent):
                        self.graph.remove_edge(agent_id, partner_id)


    def _handle_deaths_and_rebirths(self):
        dead_agent_ids_pos = []
        for agent_id, agent in self.agents.items():
            if agent.is_alive and agent.fcrit < MIN_FCRIT_SURVIVAL:
                agent.die()
                dead_agent_ids_pos.append((agent_id, agent.pos))
                if agent_id in self.graph: # Check if node exists before removing
                    self.graph.remove_node(agent_id)


        for dead_agent_id, pos in dead_agent_ids_pos:
            r, c = pos
            living_neighbors_obj = [n_agent for n_agent in self.get_neighbors(pos) if n_agent.is_alive]

            if living_neighbors_obj:
                parent_agent = random.choice(living_neighbors_obj)
                new_beta = parent_agent.beta_lever
                if random.random() < MUTATION_RATE_STRATEGY:
                    new_beta += random.uniform(-MUTATION_STRENGTH_BETA, MUTATION_STRENGTH_BETA)
            else:
                new_beta = random.uniform(0.8, 1.2)

            new_agent = Agent(dead_agent_id, pos, INITIAL_FCRIT, beta_lever=new_beta)
            self.agents[dead_agent_id] = new_agent
            self.agent_grid[r,c] = new_agent
            self.graph.add_node(dead_agent_id, agent_obj=new_agent)


    def step(self):
        self._distribute_resources()
        self._apply_link_maintenance_costs()
        self._update_network()
        self._handle_deaths_and_rebirths()
        self.current_step += 1

    def get_network_metrics(self):
        # Base case for no nodes or very few nodes
        if not self.graph.nodes() or len(self.graph.nodes()) < 2 :
            n_alive_agents_base = sum(1 for a in self.agents.values() if a.is_alive)
            avg_fcrit_val_base = np.mean([a.fcrit for a in self.agents.values() if a.is_alive]) if n_alive_agents_base > 0 else 0
            avg_pref_deg_val_base = np.mean([a.preferred_degree for a in self.agents.values() if a.is_alive]) if n_alive_agents_base > 0 else 0
            avg_beta_lever_val_base = np.mean([a.beta_lever for a in self.agents.values() if a.is_alive]) if n_alive_agents_base > 0 else 0
            return {
                "avg_fcrit": avg_fcrit_val_base, "n_alive": n_alive_agents_base,
                "avg_degree": 0, "density": 0, "avg_clustering": 0,
                "overall_avg_path_length": np.nan, "n_components": 0, "largest_comp_size":0,
                "avg_preferred_degree": avg_pref_deg_val_base, "avg_beta_lever": avg_beta_lever_val_base,
                "lcc_avg_degree": 0, "lcc_avg_path_length": np.nan, "n_living_nodes_in_lcc": 0,
                "simple_box_dim_2d": np.nan,
            }

        living_agent_ids = [id for id, agent in self.agents.items() if agent.is_alive]
        live_graph = self.graph.subgraph(living_agent_ids).copy()

        # After filtering, check again if the live_graph is too small
        if not live_graph.nodes() or len(live_graph.nodes()) < 2:
            n_alive_agents_live = sum(1 for id in living_agent_ids if self.agents[id].is_alive) # Should be len(living_agent_ids)
            avg_fcrit_val_live = np.mean([self.agents[id].fcrit for id in living_agent_ids]) if living_agent_ids else 0
            avg_pref_deg_val_live = np.mean([self.agents[id].preferred_degree for id in living_agent_ids]) if living_agent_ids else 0
            avg_beta_lever_val_live = np.mean([self.agents[id].beta_lever for id in living_agent_ids]) if living_agent_ids else 0
            return {
                "avg_fcrit": avg_fcrit_val_live, "n_alive": n_alive_agents_live,
                "avg_degree": 0, "density": 0, "avg_clustering": 0,
                "overall_avg_path_length": np.nan, "n_components": 0 if not live_graph.nodes() else 1, "largest_comp_size": len(live_graph.nodes()),
                "avg_preferred_degree": avg_pref_deg_val_live, "avg_beta_lever": avg_beta_lever_val_live,
                "lcc_avg_degree": 0, "lcc_avg_path_length": np.nan, "n_living_nodes_in_lcc": len(live_graph.nodes()),
                "simple_box_dim_2d": np.nan,
            }
        
        largest_cc_nodes = set()
        if live_graph.nodes(): # Check if live_graph has nodes before finding components
            connected_components = list(nx.connected_components(live_graph))
            if connected_components: # Check if there are any components
                largest_cc_nodes = max(connected_components, key=len)
        
        lcc_subgraph = live_graph.subgraph(largest_cc_nodes)
        lcc_avg_path = np.nan
        if len(lcc_subgraph.nodes()) >= 2:
            lcc_avg_path = nx.average_shortest_path_length(lcc_subgraph)
        
        lcc_degrees = [d for n, d in lcc_subgraph.degree()]
        lcc_avg_degree = np.mean(lcc_degrees) if lcc_degrees else 0

        avg_path = np.nan
        if live_graph.number_of_nodes() >= 2 and nx.is_connected(live_graph):
            avg_path = nx.average_shortest_path_length(live_graph)
        
        degrees = [d for n, d in live_graph.degree()]
        n_alive_val = len(living_agent_ids)
        avg_fcrit_val = np.mean([self.agents[id].fcrit for id in living_agent_ids]) if living_agent_ids else 0
        avg_pref_deg_val = np.mean([self.agents[id].preferred_degree for id in living_agent_ids]) if living_agent_ids else 0
        avg_beta_lever_val = np.mean([self.agents[id].beta_lever for id in living_agent_ids]) if living_agent_ids else 0

        metrics = {
            "avg_fcrit": avg_fcrit_val,
            "n_alive": n_alive_val,
            "avg_degree": np.mean(degrees) if degrees else 0,
            "density": nx.density(live_graph),
            "avg_clustering": nx.average_clustering(live_graph),
            "overall_avg_path_length": avg_path,
            "n_components": nx.number_connected_components(live_graph),
            "largest_comp_size": len(largest_cc_nodes),
            "avg_preferred_degree": avg_pref_deg_val,
            "avg_beta_lever": avg_beta_lever_val,
            "lcc_avg_degree": lcc_avg_degree,
            "lcc_avg_path_length": lcc_avg_path,
            "n_living_nodes_in_lcc": len(lcc_subgraph.nodes()),
            "simple_box_dim_2d": calculate_simple_box_count_2d(
                [self.agents[id].pos for id in largest_cc_nodes if id in self.agents], self.size # Ensure id exists
            ) if largest_cc_nodes else np.nan,
        }
        return metrics

    def plot_network(self, ax, title="Network Structure"):
        live_agent_ids = [id for id, agent in self.agents.items() if agent.is_alive]
        live_graph = self.graph.subgraph(live_agent_ids)

        pos = {agent_id: self.agents[agent_id].pos for agent_id in live_graph.nodes()}
        
        if not live_graph.nodes():
            ax.set_title(title + " (No living agents)")
            ax.set_xticks([])
            ax.set_yticks([])
            return

        fcrit_values = np.array([self.agents[id].fcrit for id in live_graph.nodes()])
        node_colors = fcrit_values
        
        degrees = np.array([d for n, d in live_graph.degree()])
        node_sizes = 50 + degrees * 20

        nodes_collection = nx.draw_networkx_nodes(
            live_graph,
            pos,
            node_color=node_colors,
            cmap=plt.cm.viridis_r,
            vmin=MIN_FCRIT_SURVIVAL + EPSILON,
            vmax=FCRIT_COLOR_VMAX,
            node_size=node_sizes,
            ax=ax,
        )
        nx.draw_networkx_edges(live_graph, pos, ax=ax, width=0.5, alpha=0.8)
        ax.figure.colorbar(nodes_collection, ax=ax, label="Agent Fc_crit")
        ax.set_title(title)
        ax.set_aspect('equal')


# --- Simple Box Counting Dimension (No changes needed) ---
def calculate_simple_box_count_2d(live_graph_nodes_pos, grid_size, num_box_sizes=5):
    if not live_graph_nodes_pos or len(live_graph_nodes_pos) < 2:
        return np.nan
    coords = np.array(live_graph_nodes_pos, dtype=float) / float(grid_size)
    min_box = 1.0 / grid_size
    max_box = 0.5
    if min_box >= max_box: # Prevent issues if grid_size is too small
        return np.nan
    box_sizes = np.logspace(np.log10(min_box), np.log10(max_box), num=num_box_sizes)
    counts = []
    inv_sizes = []
    for size in box_sizes:
        if size <= 0: continue # Skip invalid box sizes
        boxes = set()
        for x, y in coords:
            ix = int(np.floor(x / size))
            iy = int(np.floor(y / size))
            boxes.add((ix, iy))
        if not boxes: continue # Skip if no boxes found for this size
        counts.append(len(boxes))
        inv_sizes.append(1.0 / size)
    if len(counts) < 2:
        return np.nan
    log_counts = np.log(counts)
    log_inv_sizes = np.log(inv_sizes) # Renamed for clarity
    try:
        # Check for NaNs or Infs which can cause polyfit issues
        if np.any(np.isnan(log_inv_sizes)) or np.any(np.isinf(log_inv_sizes)) or \
           np.any(np.isnan(log_counts)) or np.any(np.isinf(log_counts)):
            return np.nan
        slope, _ = np.polyfit(log_inv_sizes, log_counts, 1)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
    return slope


# --- Main Simulation Runner (Modified for saving plots) ---
def run_simulation_regime(
    regime_type,
    sim_steps=SIMULATION_STEPS,
    report_interval=REPORT_INTERVAL,
    plot_interval=500,  # How often to save network snapshots
    run_id=None,
    save_network_snapshots=False, # Parameter to control snapshot saving
):
    run_tag = f"_{run_id}" if run_id is not None else ""
    print(f"\n--- Running Simulation for Regime: {regime_type.upper()}{run_tag} ---")
    env = MVSEnvironment(GRID_SIZE, N_AGENTS, regime_type=regime_type)
    history = collections.defaultdict(list)

    fig_network_snapshot = None # Figure for snapshots, created if needed
    if save_network_snapshots:
        fig_network_snapshot = plt.figure(figsize=(8, 8))

    # Define specific steps for which snapshots are always saved if save_network_snapshots is True
    critical_snapshot_steps = {0, sim_steps // 2, sim_steps - 1}

    for step in range(sim_steps):
        env.step()
        metrics = env.get_network_metrics()
        
        for key, value in metrics.items():
            history[key].append(value)
        history["step"].append(step)

        if step % report_interval == 0 or step == sim_steps -1:
            print(
                f"Step {step}: Alive={metrics['n_alive']}, AvgFc={metrics['avg_fcrit']:.2f}, "
                f"AvgDeg={metrics['avg_degree']:.2f}, Density={metrics['density']:.3f}, "
                f"Clustering={metrics['avg_clustering']:.3f}, PathLen={metrics['lcc_avg_path_length']:.2f}, "
                f"Components={metrics['n_components']}, LargestComp={metrics['largest_comp_size']}"
            )

        # Determine if a snapshot should be saved for this step
        should_save_this_snapshot = save_network_snapshots and \
                                   (step % plot_interval == 0 or \
                                    step == sim_steps - 1 or \
                                    step in critical_snapshot_steps)

        if should_save_this_snapshot:
            if fig_network_snapshot is None: # Should have been created if save_network_snapshots is True
                fig_network_snapshot = plt.figure(figsize=(8, 8))

            fig_network_snapshot.clf()  # Clear figure for new plot
            ax_network = fig_network_snapshot.add_subplot(111)
            env.plot_network(ax_network, title=f"{regime_type.upper()} - Step {step}")
            
            snapshot_filename = f"mvs_network_{regime_type}{run_tag}_step_{step}.png"
            fig_network_snapshot.savefig(os.path.join(RESULTS_DIR, snapshot_filename))
            # print(f"Saved network snapshot: {snapshot_filename}") # Optional: log this too

    if save_network_snapshots and fig_network_snapshot is not None:
        plt.close(fig_network_snapshot) # Close the figure after all snapshots are done

    living_nodes = [a_id for a_id, a in env.agents.items() if a.is_alive]
    final_graph = env.graph.subgraph(living_nodes)
    comp_sizes = [len(c) for c in nx.connected_components(final_graph)] if final_graph.nodes else []
    degrees = [d for _, d in final_graph.degree()] if final_graph.nodes else []

    return history, comp_sizes, degrees


if __name__ == "__main__":
    # --- Setup Results Directory and Summary Structure ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_file_path = SUMMARY_FILE

    simulation_summary = {
        "parameters": {
            "grid_size": GRID_SIZE,
            "n_agents": N_AGENTS,
            "simulation_steps": SIMULATION_STEPS,
            "report_interval": REPORT_INTERVAL,
        },
        "replications": N_ITERATIONS_PER_REGIME,
        "random_seeds": [BASE_SEED + i for i in range(N_ITERATIONS_PER_REGIME)],
        "regimes": {},
    }

    # --- Simulation ---
    print(f"Simulation started. Summary will be saved to: {log_file_path}")
    print(f"All plots will be saved in the '{RESULTS_DIR}' directory.\n")

    regimes_to_run = ["scarcity", "shock", "flux"]

    all_histories = {}

    for regime_idx, regime in enumerate(regimes_to_run):
        run_histories = []
        aggregated_comp_sizes = []
        aggregated_degrees = []
        
        print(f"\n===== PROCESSING REGIME: {regime.upper()} =====")
        for it in range(N_ITERATIONS_PER_REGIME):
            print(f"\n--- Iteration {it+1}/{N_ITERATIONS_PER_REGIME} for {regime.upper()} ---")
            seed = BASE_SEED + it # Vary seed for each iteration
            random.seed(seed)
            np.random.seed(seed)

            history, comp_sizes, degrees = run_simulation_regime(
                regime,
                sim_steps=SIMULATION_STEPS,
                report_interval=REPORT_INTERVAL,
                plot_interval=SIMULATION_STEPS // 4, # Interval for saving snapshots
                run_id=f"run{it}",
                save_network_snapshots=(it == 0), # Only save snapshots for the first run of each regime
            )

            run_histories.append(history)
            aggregated_comp_sizes.extend(comp_sizes)
            aggregated_degrees.extend(degrees)

        all_histories[regime] = run_histories

        # ----- Averaged metrics plots -----
        print(f"\nGenerating averaged metrics plot for {regime.upper()}...")
        plot_keys = [
            "n_alive", "avg_fcrit", "avg_degree", "avg_clustering",
            "lcc_avg_path_length", "avg_preferred_degree",
            "avg_beta_lever", "simple_box_dim_2d",
        ]
        
        if not run_histories:
            print(f"No run histories for {regime}, skipping metrics plot.")
            continue
        
        steps = run_histories[0]["step"] # Assumes all runs have the same number of steps
        num_plots = len(plot_keys)
        
        fig_metrics, axes_metrics = plt.subplots(num_plots, 1, figsize=(12, 2.5 * num_plots), sharex=True)
        if num_plots == 1: axes_metrics = [axes_metrics] # Ensure it's a list

        for idx, key in enumerate(plot_keys):
            # Collect data, handling cases where a key might be missing or data is empty
            data_for_key = [h[key] for h in run_histories if key in h and h[key]]
            if not data_for_key:
                print(f"Warning: No data for metric '{key}' in regime '{regime}'. Plotting placeholder.")
                axes_metrics[idx].text(0.5, 0.5, f"No data for {key}", horizontalalignment='center', verticalalignment='center', transform=axes_metrics[idx].transAxes)
                axes_metrics[idx].set_ylabel(key.replace("_", " ").title())
                axes_metrics[idx].grid(True, linestyle="--", alpha=0.7)
                continue

            # Ensure all data arrays for this key have the same length as 'steps' for proper aggregation
            # This typically means padding if some runs ended prematurely or had varying step counts (shouldn't happen here)
            max_len = len(steps)
            processed_data = []
            for arr in data_for_key:
                if len(arr) < max_len:
                    # Pad with the last valid value or NaN
                    pad_value = arr[-1] if arr else np.nan
                    processed_data.append(np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=pad_value))
                elif len(arr) > max_len:
                    processed_data.append(arr[:max_len]) # Truncate if too long
                else:
                    processed_data.append(arr)
            
            data_np = np.array(processed_data)
            mean = np.nanmean(data_np, axis=0)
            std = np.nanstd(data_np, axis=0)

            axes_metrics[idx].plot(steps, mean, label=key)
            axes_metrics[idx].fill_between(steps, mean - std, mean + std, alpha=0.2)
            axes_metrics[idx].set_ylabel(key.replace("_", " ").title())
            axes_metrics[idx].grid(True, linestyle="--", alpha=0.7)
            axes_metrics[idx].legend()

        axes_metrics[-1].set_xlabel("Simulation Step")
        fig_metrics.suptitle(
            f"Averaged Network Metrics Evolution - {regime.upper()} Regime (N={N_ITERATIONS_PER_REGIME} runs)"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        metrics_plot_filename = f"mvs_metrics_{regime}.png"
        fig_metrics.savefig(os.path.join(RESULTS_DIR, metrics_plot_filename))
        plt.close(fig_metrics) # Close the figure to free memory
        print(f"Saved averaged metrics plot: {metrics_plot_filename}")


        # ----- Aggregated distributions -----
        print(f"Generating distribution plots for {regime.upper()}...")
        # Component Size Distribution
        if aggregated_comp_sizes:
            fig_comp, ax_comp = plt.subplots() # Create new figure and axes
            ax_comp.hist(
                aggregated_comp_sizes,
                bins=range(1, max(aggregated_comp_sizes) + 2) if aggregated_comp_sizes else [0,1],
                edgecolor="black",
            )
            ax_comp.set_xlabel("Component Size")
            ax_comp.set_ylabel("Count")
            ax_comp.set_title(
                f"Aggregated Component Size Distribution - {regime.upper()} (N={N_ITERATIONS_PER_REGIME} runs)"
            )
            comp_dist_filename = f"mvs_component_size_dist_{regime}.png"
            fig_comp.savefig(os.path.join(RESULTS_DIR, comp_dist_filename))
            plt.close(fig_comp) # Close the figure
            print(f"Saved component size distribution: {comp_dist_filename}")
        else:
            print(f"No component size data to plot for {regime.upper()}.")


        # Degree Distribution
        if aggregated_degrees:
            fig_deg, ax_deg = plt.subplots() # Create new figure and axes
            ax_deg.hist(
                aggregated_degrees,
                bins=range(0, max(aggregated_degrees) + 2) if aggregated_degrees else [0,1],
                edgecolor="black",
            )
            ax_deg.set_xlabel("Node Degree")
            ax_deg.set_ylabel("Count")
            ax_deg.set_title(
                f"Aggregated Degree Distribution - {regime.upper()} (N={N_ITERATIONS_PER_REGIME} runs)"
            )
            deg_dist_filename = f"mvs_degree_dist_{regime}.png"
            fig_deg.savefig(os.path.join(RESULTS_DIR, deg_dist_filename))
            plt.close(fig_deg) # Close the figure
            print(f"Saved degree distribution: {deg_dist_filename}")
        else:
            print(f"No degree data to plot for {regime.upper()}.")

        print(f"Finished processing for regime: {regime.upper()}")

        # --- Collect summary information for this regime ---
        final_iteration = {}
        if run_histories:
            metric_keys = [k for k in run_histories[0] if k != "step"]
            for k in metric_keys:
                vals = [h[k][-1] for h in run_histories]
                final_iteration[k] = {
                    "mean": float(np.nanmean(vals)),
                    "std": float(np.nanstd(vals)),
                }
            final_iteration.pop("n_living_nodes_in_lcc", None)

        regime_summary = {
            "final_iteration": final_iteration,
        }

        # --- Condensed time series information ---
        timeseries_keys = [
            "avg_fcrit",
            "avg_degree",
            "avg_clustering",
            "lcc_avg_path_length",
            "simple_box_dim_2d",
        ]
        time_series = {}
        for key in timeseries_keys:
            data_for_key = [h[key] for h in run_histories if key in h]
            if not data_for_key:
                continue
            data_np = np.array(data_for_key)
            mean_series = np.nanmean(data_np, axis=0)
            std_series = np.nanstd(data_np, axis=0)
            if mean_series.size == 0:
                continue
            slope = (mean_series[-1] - mean_series[0]) / max(len(mean_series) - 1, 1)
            time_series[key] = {
                "start": float(mean_series[0]),
                "final": float(mean_series[-1]),
                "min": float(np.nanmin(mean_series)),
                "max": float(np.nanmax(mean_series)),
                "slope_per_step": float(slope),
                "final_std": float(std_series[-1]),
            }
        regime_summary["time_series"] = time_series

        # --- Final distribution summary ---
        degree_array = np.array(aggregated_degrees)
        comp_array = np.array(aggregated_comp_sizes)
        final_dists = {
            "degree": {
                "mean": float(np.mean(degree_array)) if degree_array.size else 0,
                "var": float(np.var(degree_array)) if degree_array.size else 0,
                "min": int(degree_array.min()) if degree_array.size else 0,
                "max": int(degree_array.max()) if degree_array.size else 0,
            },
            "comp_size": {
                "n_components": int(comp_array.size),
                "largest_comp_size": int(comp_array.max()) if comp_array.size else 0,
                "mean": float(np.mean(comp_array)) if comp_array.size else 0,
                "var": float(np.var(comp_array)) if comp_array.size else 0,
            },
        }
        if final_iteration:
            lcs = final_iteration.get("largest_comp_size", {}).get("mean", 0)
            na = final_iteration.get("n_alive", {}).get("mean", 1)
            ratio = lcs / na if na else 0
            final_dists["comp_size"]["p_in_lcc"] = float(ratio)
        regime_summary["final_distributions"] = final_dists

        # --- Resource schedule information ---
        if regime == "scarcity":
            regime_summary["resource_schedule"] = {
                "type": "constant",
                "inflow_per_step": SCARCITY_INFLOW,
            }
        elif regime == "shock":
            regime_summary["resource_schedule"] = {
                "type": "periodic_shock",
                "base_inflow": SHOCK_BASE_INFLOW,
                "reduced_inflow": SHOCK_REDUCED_INFLOW,
                "shock_duration": SHOCK_DURATION,
                "shock_interval": SHOCK_INTERVAL,
            }
        elif regime == "flux":
            regime_summary["resource_schedule"] = {
                "type": "moving_sources",
                "base_inflow": FLUX_BASE_INFLOW,
                "n_sources": FLUX_N_SOURCES,
                "source_lifespan": FLUX_SOURCE_LIFESPAN,
            }

        simulation_summary["regimes"][regime] = regime_summary

    print("\nSimulation Complete. All output and plots saved in 'results' directory.")

    # --- Save summary information to JSON ---
    with open(log_file_path, "w") as f:
        f.write(PREAMBLE_TEXT.strip() + "\n")
        json.dump(simulation_summary, f, indent=2)
    print(f"Simulation summary saved to {log_file_path}")

    # Final message to console
    print(f"\nMain simulation script finished. Check the '{RESULTS_DIR}' folder for all outputs.")
    # plt.ioff() # REMOVED - No interactive mode was turned on
