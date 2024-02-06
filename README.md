<!--
 * @Author: WANG Maonan
 * @Date: 2023-09-15 16:46:26
 * @Description: LA-Light README
 * @LastEditTime: 2024-02-05 17:49:10
-->
# LLM-Assisted Light (LA-Light)

LLM-Assisted Light: Augmenting Traffic Signal Control with Large Language Model in Complex Urban Scenarios

<div align="center">
  <a href="https://github.com/Traffic-Alpha/LLM-Assisted-Light/assets/21176109/e7063493-30a7-4f12-95b5-6b0661e48f19">
    <img src="https://github.com/Traffic-Alpha/LLM-Assisted-Light/assets/21176109/e7063493-30a7-4f12-95b5-6b0661e48f19" width="80%" alt="视频链接">
  </a>
  <p>123</p>
</div>


## Overall Framework

The LA-Light framework introduces an innovative hybrid decision-making process for TSC that leverages the cognitive capabilities of LLMs alongside traditional traffic management methodologies. This framework includes **five methodical steps** for decision-making: 
- **Step 1** outlines the task planning phase where the LLM defines its role in traffic management. 
- **Step 2** involves the selection of appropriate perception and decision-making tools by the LLM. 
- **Step 3** utilizes these tools interact with the traffic environment to gather data. 
- **Step 4** depicts the analysis of this data by the Decision Unit to inform decision-making. 
- **Step 5** illustrates the implementation of the LLM's decisions and the provision of explanatory feedback for system transparency and validation

<div align=center>
  <img width="90%" src="./assets/framework.png" />
</div>


## Evaluating LA-Light

### Training and Evaluating the RL Model

For training and evaluating the RL model, refer to [TSCRL](./TSCRL/). You can use the following command to start training:

```shell
python train_rl_agent.py
```

The [RL Result](./TSCRL/result/) directory contains the trained models and training results. Use the following command to evaluate the performance of the model:

```shell
python eval_rl_agent.py
```

### Pure LLM

To directly use LLM for inference without invoking any tools, run the following script:

```shell
python llm.py --env_name '3way' --phase_num 3 --detector_break 'E0--s'
```

### Decision Making with LLM + RL

To test LA-Light, run the following script. In this case, we will randomly generate congestion on `E1` and the sensor on the `E2--s` direction will fail.

```shell
python llm_rl.py --env_name '4way' --phase_num 4 --edge_block 'E1' --detector_break 'E2--s'
```

The effect of running the above test is shown in the following video. Each decision made by LA-Light involves multiple tool invocations and subsequent decisions based on the tool's return results, culminating in a final decision and explanation.

[LLM_for_TSC_README.webm](https://github.com/Traffic-Alpha/LLM-Assisted-Light/assets/21176109/131281d9-831d-4e08-919c-2ee8ac3fd841)

Due to the video length limit, we only captured part of the first decision-making process, including:

- Action 1: Obtaining the intersection layout, the number of lanes, and lane functions (turn left, go straight, or turn right) for each edge.
- Action 3: Obtaining the occupancy of each edge. The -E3 straight line has a higher occupancy rate, corresponding to the simulation. At this point, LA-Light can use tools to obtain real-time road network information.
- Final Decision and Explanation: Based on a series of results, LA-Light provides the final decision and explanation.

## Acknowledgments

We would like to thank the authors and developers of the following projects, this project is built upon these great open-sourced projects.
- [TransSimHub](https://github.com/Traffic-Alpha/TransSimHub)
- [LangChain](https://github.com/hwchase17/langchain)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

## Contact

- If you have any questions, please report issues on GitHub.
