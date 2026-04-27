# Longevity Society Simulation – LLM Hackathon

This project contains a baseline configuration and high‑level design for running a simple agent‑based simulation using large language models (LLMs).  The goal of the simulation is to explore how human behaviour might change in a future where the average lifespan extends to around **200 years**, youthfulness is maintained into old age, and reproductive capability spans well beyond the current human norm.

The simulation is derived from the bar–fire example in the original LLM‑Hackathon starter code.  Agents move on a two‑dimensional grid, communicate with nearby agents via LLM‑generated messages, make decisions about where to go next and how to react to events, and remember past observations.  In this version, the **places** and **events** have been redefined to represent schools, workplaces, family areas, policy debates and other social institutions relevant to a longevity‑oriented society.

## Project structure

```
LLM-hackathon-project/
├── README.md             # You are here
├── config.yaml           # YAML configuration file for the simulation
└── (placeholder files)   # You can add simulation code (e.g. agent.py, simulation.py) here
```

Only the configuration and design document are provided here.  The actual simulation engine (such as `simulation.py` and `agent.py`) should be taken from the original hackathon repository or implemented independently.  The config file below can be used with the existing simulation code to run the longevity scenario.

## Longevity scenario

The core assumptions of this scenario are:

* **Lifespan**: Humans live up to ~200 years on average.  They remain physically youthful (appearing around 30 years old) until roughly 180 years of age.
* **Reproductive window**: People can reproduce from their late teens through roughly 150 years of age.
* **Retirement**: There is no compulsory retirement age; people may work, learn and change careers at any time.
* **Long‑term planning**: Agents may prioritise long‑term goals over short‑term gains because their time horizons are much longer.
* **Events**: Instead of fires, “events” represent social or policy changes—e.g. pension reforms, debates about population control, or sudden increases in the cost of rejuvenation treatments.

Agents are given additional attributes such as age, wealth, family orientation, risk tolerance and desire for children.  They decide where to go—academic institutions, family zones, startup laboratories, rejuvenation clinics, governance centres or investment hubs—based on their attributes, current events and interactions with others.

## Running the simulation

1. Clone this repository together with the original hackathon code.
2. Make sure you have Python 3.10+ and required dependencies installed (e.g. `openai`, `numpy`, `scipy`, etc.).
3. Replace or extend the existing `config.yaml` with the one provided here.
4. Run the simulation script (e.g. `python simulation.py --config config.yaml`).  The script will read the configuration, initialise agents and places, and run the simulation for the specified duration.  Output data (agent trajectories, messages and memory states) will be saved to the `output_longevity` directory.

You are encouraged to modify the configuration, add new places or events, and experiment with different parameters to observe emergent behaviours.  Feel free to extend the agent model to include more nuanced decision logic, add new KPIs (e.g. marriage age, number of children, career changes) and analyse how they evolve over time.

## Next steps

To fully realise this scenario, you may want to:

* **Integrate the society settings into your agent code** so that agents are aware of their age, wealth and other attributes when making decisions.
* **Define KPI trackers** in the simulation to measure outcomes such as marriage age, number of children, career changes, education events and wealth distribution.
* **Visualise the results** using charts or animations to compare longevity societies against baseline (modern) societies.

Contributions and extensions are welcome!  Fork this repository and experiment with different societal assumptions or agent behaviours, and see what emergent phenomena arise.