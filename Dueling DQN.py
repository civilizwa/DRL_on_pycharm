import random

import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import rl_utils
import matplotlib.pyplot as plt
import safety_gymnasium

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)

    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        #这里采用小样本等概率随机采样
        transitions=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done

    def size(self):
        return len(self.buffer)

class VAnet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(VAnet,self).__init__()
        #共享网络部分，可换成卷积层提取特征
        self.fc1=nn.Linear(state_dim,hidden_dim)
        #A网络部分
        self.fcA=nn.Linear(hidden_dim,action_dim)
        #V网络部分
        self.fcV=nn.Linear(hidden_dim,1)

    def forward(self,x):
        A=self.fcA(F.relu(self.fc1(x)))
        V=self.fcV(F.relu(self.fc1(x)))
        Q=V+A-A.mean(1).view(-1,1)
        return Q

"""在基础DQN上实现double结合Dueling的网络
double:更新VAnet时，loss=q-y,y：先利用VAnet网络获取action,再将action带入targetNet获取最大值
dueling:改变原来qnet的结构为VAnet"""
class DQN:
    def __init__(self,
                 state_dim,
                 hidden_dim,
                 action_dim,
                 lr,
                 gamma,
                 epsilon,
                 target_update,
                 device):
        self.action_dim=action_dim
        self.lr=lr
        #Q-network
        self.q_net=VAnet(state_dim,hidden_dim,action_dim).to(device)
        #Target-network
        self.target_net=VAnet(state_dim,hidden_dim,action_dim).to(device)

        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=self.lr)

        self.gamma=gamma
        self.epsilon=epsilon#采用贪婪策略，减弱自举行为
        self.target_update=target_update#target网络更新频率
        self.count=0 #记录更新次数
        self.device=device

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.action_dim)
        else:
            state=torch.tensor([state],dtype=torch.float).to(self.device)
            action=self.q_net(state).argmax().item()
        return action

    def update(self,transition_dict):
        states=torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions=torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict['rewards']).view(-1,1).to(self.device)
        next_states=torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones=torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)

        q_values=self.q_net(states).gather(1,actions)
        #该部分与DQN不同，是double-DQN的实现
        """在q_net上选择下一个状态的动作"""
        """max(1)[1]:选取最大Q值对应的索引——即动作"""
        max_actions=self.q_net(next_states).max(1)[1].view(-1,1)
        """在target_net上获得该动作对应的价值"""
        max_next_q_values=self.target_net(next_states).gather(1,max_actions)
        q_targets=rewards+self.gamma*max_next_q_values*(1-dones)
        dqn_loss=torch.mean(F.mse_loss(q_values,q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        #固定频率更新target网络
        if self.count%self.target_update==0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        self.count+=1

lr=2e-3
num_episodes=500
hidden_dim=128
gamma=0.98
epsilon=0.01
target_update=10
buffer_size=10000
minimal_size=500
batch_size=64
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name='SafetyPointGoal0-v0'
env=safety_gymnasium.make(env_name)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

replay_buffer=ReplayBuffer(buffer_size)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n
agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device)

return_list=[]
convergence_list = []
# 设定目标收敛奖励
target_reward = 195

for i in range(10):
    with tqdm(total=int(num_episodes/10),desc='Iteration %d'%i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return=0
            state=env.reset()
            done=False
            while not done:
                #env.render()
                action=agent.take_action(state)
                next_state,reward,done,_=env.step(action)
                replay_buffer.add(state,action,reward,next_state,done)
                state=next_state
                episode_return+=reward
                if replay_buffer.size()>minimal_size:
                    b_s,b_a,b_r,b_ns,b_d=replay_buffer.sample(batch_size)
                    transition_dict={
                        'states':b_s,
                        'actions':b_a,
                        'rewards':b_r,
                        'next_states':b_ns,
                        'dones':b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            #计算并记录收敛速度
            convergence = rl_utils.calculate_convergence_speed(return_list)
            convergence_list.append(convergence)
            if (i_episode+1)%10==0:
                pbar.set_postfix({'episode':'%d'%(num_episodes/10*i+i_episode+1),'return':'%.3f'%np.mean(return_list[-10:])})
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Double-Dueling-DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Double-Dueling-DQN on {}'.format(env_name))
plt.show()

# 绘制收敛速度图像
plt.plot(convergence)
plt.title('Convergence plot')
plt.xlabel('Episode')
plt.ylabel('Convergence (in number of episodes)')
plt.axhline(y=100, linestyle='--', color='red') # 添加目标收敛速度线
plt.show()