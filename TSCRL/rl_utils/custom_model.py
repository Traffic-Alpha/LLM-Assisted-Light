'''
@Author: WANG Maonan
@Date: 2023-09-08 18:34:24
@Description: Custom Model
@LastEditTime: 2023-11-27 23:03:32
'''
import gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomTSCModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 16):
        """特征提取网络, (7,12) 由两个部分组成:
        1. (5,12), 每个 movement 的占有率
            -> embedding, (B,5,12) -> (B,5,32)
            -> lstm, (B,5,32) -> (B,64)
        2. (2,12), phase 的信息
            -> phase_embedding, (B,2,12) -> (B,2,16)
            -> flatten, (B,2,16) -> (B,32)
        3. 合并 occ 和 phase 信息
            -> concat, (B,64) + (B,32) -> (B,96)
            -> output, (B,96) -> (B,32) -> (B,16)
        """
        super().__init__(observation_space, features_dim)
        net_shape = observation_space.shape[-1] # 12

        # 1, occ
        self.occ_embedding = nn.Sequential(
            nn.Linear(net_shape, 32),
            nn.ReLU(),
        ) # 5*12 -> 5*32
        self.occ_lstm = nn.LSTM(
            input_size=32, hidden_size=64,
            num_layers=1, batch_first=True
        )
        self.relu = nn.ReLU()

        # 2, phase index
        self.phase_embedding = nn.Sequential(
            nn.Linear(net_shape, 16),
            nn.ReLU(),
        ) # 2*12 -> 2*16

        # 3, concat and output
        self.output = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, features_dim)
        )

    def forward(self, observations):
        # Extract occupancy (occ) and phase information from x
        occ = observations[:, :5, :]  # Assuming x is (B, 7, 12) and occ is (B, 5, 12)
        phase = observations[:, 5:, :]  # Assuming phase is (B, 2, 12)
        B = occ.size(0) # 获得 batch_size 大小
        
        # Process occupancy information
        occ_embedded = self.occ_embedding(occ).view(B, 5, -1)
        _, (occ_lstm_out, _)  = self.occ_lstm(occ_embedded)
        occ_lstm_out = occ_lstm_out[-1] # (B,64)
        occ_vector = self.relu(occ_lstm_out)

        # Process phase information
        phase_embedded = self.phase_embedding(phase).view(B, 2, -1)  # (B, 2, 16)
        phase_flattened = phase_embedded.view(B, -1)  # Flatten to (B, 32)

        # Concatenate occ and phase information
        combined = torch.cat((occ_vector, phase_flattened), dim=1)  # (B, 96)
        output = self.output(combined)

        return output