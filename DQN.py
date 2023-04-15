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

class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)

    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        #这里采用小样本等概率随机采样
        transitions=random.sample(self.buffer,batch_size)
        """state,action,reward,next_state,done=zip(*transition)执行后，将
        (s1, a1, r1, s2, d1), (s2, a2, r2, s3, d2), (s3, a3, r3, s4, d3)
        转换为
        state = (s1, s2, s3)
        action = (a1, a2, a3)
        reward = (r1, r2, r3)
        next_state = (s2, s3, s4)
        done = (d1, d2, d3)
        """
        state,action,reward,next_state,done=zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done

    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Qnet,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,action_dim)

    def forward(self,x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)

"""这里使用的DQN是基于target-net的实现，参考wss视频3.2的Target Network"""
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device):
        self.action_dim=action_dim
        self.lr=lr
        #Q-network
        self.q_net=Qnet(state_dim,hidden_dim,action_dim).to(device)
        #Target-network
        self.target_net=Qnet(state_dim,hidden_dim,action_dim).to(device)

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
        """q_net(states)返回(batch_size,actions_num)的张量，
        gather(1,actions)用于获取对应action的value
        因此返回结果为(batch_size,1)"""
        q_values=self.q_net(states).gather(1,actions)
        max_next_q_values=self.target_net(next_states).max(1)[0].view(-1,1)
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

env_name='CartPole-v0'
env=gym.make(env_name)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

replay_buffer=ReplayBuffer(buffer_size)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n
agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device)

return_list=[]
for i in range(10):
    with tqdm(total=int(num_episodes/10),desc='Iteration %d'%i) as pbar:
        for i_episode in range(int(num_episodes/10)):
            episode_return=0
            state=env.reset()
            done=False
            while not done:
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
            if (i_episode+1)%10==0:
                pbar.set_postfix({'episode':'%d'%(num_episodes/10*i+i_episode+1),'return':'%.3f'%np.mean(return_list[-10:])})
            pbar.update(1)

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()
