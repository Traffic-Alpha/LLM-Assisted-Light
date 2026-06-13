<!--
 * @Author: WANG Maonan
 * @Date: 2023-09-15 16:46:26
 * @Description: LA-Light README
 * @LastEditTime: 2026-06-13 23:59:47
-->
# 🚦 LLM-Assisted Light (LA-Light)

[![arXiv](https://img.shields.io/badge/arXiv-2403.08337-b31b1b.svg)](https://arxiv.org/abs/2403.08337)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
![Version](https://img.shields.io/badge/version-1.0.0-green)

Official implementation of [LLM-Assisted Light: Augmenting Traffic Signal Control with Large Language Model in Complex Urban Scenarios](https://arxiv.org/abs/2403.08337).

## 📢 Latest News
- **[June 2026]** 🔧 **Codebase refresh**: dynamic special events (accidents, special vehicles, sensor failures) are now loaded from a YAML config (`--event-config`) instead of being hard-coded; the LLM agent no longer depends on LangChain (lightweight OpenAI-based ReAct); and static intersection info (layout, phase structure, available actions) is injected into the prompt once per episode, simplifying tool calls to dynamic state only.
- **[September 2025]** 🎉 **VLMLight accepted at NeurIPS 2025!** Congratulations! Our VLM-based traffic signal control paper has been accepted at NeurIPS 2025. [Paper Link](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3849b5861dcaeaf4758eef0979a98cc6-Abstract-Conference.html)
- **[July 2025]** **Introducing [VLMLight](https://github.com/Traffic-Alpha/VLMLight)**: Our next-generation framework featuring **image-based traffic signal control** using Vision-Language Models (VLMs) for enhanced scene understanding and real-time decision-making.
- **[August 2023]** We have migrated the simulation platform used in this project from Aiolos to [TransSimHub](https://github.com/Traffic-Alpha/TransSimHub) (TSHub). We would like to express our sincere gratitude to our colleagues at SenseTime, **@KanYuheng (阚宇衡)**, **@MaZian (马子安)**, and **@XuChengcheng (徐承成)** (in alphabetical order) for their valuable contributions. The development of TransSimHub (TSHub) is a continuation of the work done on Aiolos.


## 🧩 Core Framework

Five-stage hybrid decision-making for human-AI collaborative traffic control:
1. **Task Planning**: LLM defines traffic management role
2. **Tool Selection**: Dynamically invokes perception & decision tools
3. **Environment Interaction**: Real-time traffic data collection
4. **Data Analysis**: Decision unit generates control strategies
5. **Execution Feedback**: Implements decisions with explainable justifications

<div align=center>
  <img width="90%" src="./assets/framework.png" />
</div>

## 🧪 Quick Validation

The current lightweight validation code keeps the LA-Light tool-calling idea and runs on single-intersection TSHub/SUMO scenarios with configurable traffic events.

### 🛠️ Installation

Install [TransSimHub](https://github.com/Traffic-Alpha/TransSimHub):

```bash
git clone https://github.com/Traffic-Alpha/TransSimHub.git
cd TransSimHub
pip install -e ".[all]"
```

Install the small dependency set used by this repository:

```bash
pip install openai pyyaml loguru gymnasium numpy pytest
```

Copy and edit the local configuration file:

```bash
cp config.yaml.example config.yaml
```

Set `OPENAI_API_KEY` in `config.yaml` or via environment variable. The local `config.yaml` is ignored by git.

### 🤖 LLM-Assisted Traffic Signal Control

Run the LLM-based traffic signal controller:

```bash
python run_llm_tsc.py \
  --scenario 4way \
  --phase-num 4 \
  --event-config scenarios/4way/events/accident_set1.yaml
```

The current controller uses the LLM agent for every traffic-signal decision step. It queries tools for intersection layout, signal phases, current occupancy, traditional max-pressure recommendation, emergency vehicles, accident details, blocked movements, and sensor failures.

Optional decision logs:

```bash
python run_llm_tsc.py \
  --scenario 4way \
  --phase-num 4 \
  --event-config scenarios/4way/events/accident_set1.yaml \
  --decision-log /tmp/llm_decision.jsonl \
  --raw-response-log /tmp/llm_raw.jsonl
```

`--decision-log` stores structured decision records. `--raw-response-log` stores raw LLM outputs separately for debugging.

### 📊 Max-Pressure Baseline

Run the max-pressure baseline on the same scenario and event configuration:

```bash
python run_maxpressure.py \
  --scenario 4way \
  --phase-num 4 \
  --event-config scenarios/4way/events/accident_set1.yaml \
  --decision-log /tmp/maxpressure_decision.jsonl
```

The baseline uses the environment's `get_traditional_decision()` recommendation, so it is aligned with the traditional-decision tool seen by the LLM agent.

### 🚑 Event-Aware Evaluation

Events are configured in:

```text
scenarios/<scenario>/events/*.yaml
```

Supported event types:

- `ACCIDENTS`: insert stopped obstacle vehicles and expose affected movements
- `SPECIAL_VEHICLES`: schedule ambulance/police/rescue vehicles
- `SENSOR_FAILURES`: mask detector/movement occupancy

Compare tripinfo results, including both regular traffic and special-vehicle efficiency:

```bash
python analyze_tsc_results.py \
  llm:4way_llm_tsc.tripinfo.xml \
  maxpressure:4way_maxpressure.tripinfo.xml
```

The report separates:

- `regular`: ordinary traffic efficiency
- `special`: ambulance/rescue/police/fire vehicle completion rate, waiting time, and time loss
- `all`: aggregate metrics

See [docs/event_config.md](docs/event_config.md) for event configuration details.

### 🧪 Tests

```bash
python -m pytest tests
```

### Notes on the Current Lightweight Code

The original project included RL training/evaluation and legacy launchers. The current runnable validation path is centered on:

- `run_llm_tsc.py`: event-aware LLM controller with local tool-calling agent
- `run_maxpressure.py`: aligned max-pressure baseline
- `analyze_tsc_results.py`: regular/special vehicle metric comparison

RL training and trained models are not included in this lightweight validation path.

## 🎥 LA-Light Joint Decision-Making Demo

The following video shows the original LA-Light decision-making process. Each decision involves multiple tool invocations and subsequent reasoning based on tool-returned observations, culminating in a final decision and explanation.

[LLM_for_TSC_README.webm](https://github.com/Traffic-Alpha/LLM-Assisted-Light/assets/21176109/131281d9-831d-4e08-919c-2ee8ac3fd841)

Due to the video length limit, we only captured part of the first decision-making process, including:

- Action 1: Obtaining the intersection layout, the number of lanes, and lane functions (turn left, go straight, or turn right) for each edge.
- Action 3: Obtaining the occupancy of each edge. The -E3 straight line has a higher occupancy rate, corresponding to the simulation. At this point, LA-Light can use tools to obtain real-time road network information.
- Final Decision and Explanation: Based on a series of results, LA-Light provides the final decision and explanation.

## 🎥 Scenario Demos

[Scenario_1](https://github.com/Traffic-Alpha/LLM-Assisted-Light/assets/21176109/3075d18c-a6eb-4b5c-bdc9-f79936e13dc2)
<p align="center">Examples of LA-Lights Utilizing Tools to Control Traffic Signals <strong>(Normal Scenario)</strong></p>

[Scenario_2](https://github.com/Traffic-Alpha/LLM-Assisted-Light/assets/21176109/9062f888-314d-43f8-b668-9ad46471504c)
<p align="center">Examples of LA-Lights Utilizing Tools to Control Traffic Signals <strong>(Emergency Vehicle (EMV) Scenario)</strong></p>

## 📜 Citation

If you find this work useful, please cite our papers:

```bibtex
@article{wang2024llm,
  title={LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments},
  author={Wang, Maonan and Pang, Aoyu and Kan, Yuheng and Pun, Man-On and Chen, Chung Shue and Huang, Bo},
  journal={arXiv preprint arXiv:2403.08337},
  year={2024}
}

@inproceedings{wang2025vlmlight,
 author = {Wang, Maonan and Chen, Yirong and Pang, Aoyu and Cai, Yuxin and Chen, Chung Shue and Kan, Yuheng and Pun, Man On},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {39590--39621},
 publisher = {Curran Associates, Inc.},
 title = {{VLMLight}: Safety-Critical Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning Architecture},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/3849b5861dcaeaf4758eef0979a98cc6-Paper-Conference.pdf},
 volume = {38},
 year = {2025}
}

@ARTICLE{pang2026illm,
  author={Pang, Aoyu and Wang, Maonan and Pun, Man-On and Chen, Chung Shue and Xiong, Xi},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={{iLLM-TSC}: Integration Reinforcement Learning and Large Language Model for Traffic Signal Control Policy Improvement}, 
  year={2026},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TVT.2026.3674284}
}
```

You may also be interested in our earlier work on RL-based traffic signal control (TSC):

```bibtex
@ARTICLE{wang2024unitsa,
  author={Wang, Maonan and Xiong, Xi and Kan, Yuheng and Xu, Chengcheng and Pun, Man-On},
  journal={IEEE Transactions on Vehicular Technology}, 
  title={UniTSA: A Universal Reinforcement Learning Framework for V2X Traffic Signal Control}, 
  year={2024},
  volume={73},
  number={10},
  pages={14354-14369},
  doi={10.1109/TVT.2024.3403879}
}

@ARTICLE{wang2024ccda,
  author={Wang, Maonan and Chen, Yirong and Kan, Yuheng and Xu, Chengcheng and Lepech, Michael and Pun, Man-On and Xiong, Xi},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={Traffic Signal Cycle Control With Centralized Critic and Decentralized Actors Under Varying Intervention Frequencies}, 
  year={2024},
  volume={25},
  number={12},
  pages={20085-20104},
  doi={10.1109/TITS.2024.3462153}
}
```

## 🤝 Open-Source Foundations

This project stands on the shoulders of these open-source giants:
- [TransSimHub](https://github.com/Traffic-Alpha/TransSimHub)
- [LangChain](https://github.com/hwchase17/langchain)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

## 📮 Contact

If you have any questions, please report issues on GitHub.
