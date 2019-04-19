import os
import random
import time
import platform
import torch
import gym
import numpy as np
from collections import deque
from torch.autograd import Variable


class Policy(torch.nn.Module):
    """
    We implement a 3-layer convolutional network for a specific agent
    identified by agent_inx. We comment out the LSTM implementation for now!
    """

    def __init__(self, input_channels, num_actions, agent_idx=1):
        super(Policy, self).__init__()
        
        # Tribal parameters
        self.tribe = None
        self.color = None
        self.idx = agent_idx   # This allows multiple learning agents       
        
        # laser parameters
        self.tagged = False        
        self.laser_fired = False
        self.US_hit = 0
        self.THEM_hit = 0

        # zone parameters
        self.in_banned = False  # 03-02-2019
        self.in_target = False
        
        self.temperature = 1.0               # This is to adjust exploit/explore 
        self.input_channels = input_channels
        self.num_actions = num_actions
        
        self.features = self._init_features()
        self.action_head = self._init_action_head()
        
        # Deactivate actor-critic (CNN-LSTM) for now
        # self.lstm = self._init_lstm()
        # self.action_head = self._init_action_head()
        # self.value_head = self._init_value_head()

        # episode history
        self.saved_actions = []
        self.rewards = []
        self.log_probs = []   # Added to implement REINFORCE for PyTorch 0.4.1
        self.tagged_hist = []       
        self.tag_hist = []
        self.US_hits = []
        self.THEM_hits = []
        self.in_banned_hist = []    # 03-02-2019
        self.in_target_hist = []
        

    def _init_features(self):
        
        layers = []
        
        # [1,input_channels,10,20] input 3D array
        layers.append(torch.nn.Conv2d(self.input_channels,
                                      16, kernel_size=3, stride=1, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,10,20] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=4, stride=2, padding=1))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,5,10] feature maps
        layers.append(torch.nn.Conv2d(16,
                                      16, kernel_size=3, stride=1, padding=0))
        layers.append(torch.nn.BatchNorm2d(16))
        layers.append(torch.nn.ReLU(inplace=True))
        # [1,16,3,8] feature maps

        return torch.nn.Sequential(*layers)

    def _init_action_head(self):
        # input [1,384]
        return torch.nn.Linear(384, self.num_actions)   # output [1,8]
    
    """
    # Disable CNN-LSTM actor critic for now

    def _init_lstm(self):
        return torch.nn.LSTMCell(32*4*4, 256)

    def _init_action_head(self):
        return torch.nn.Linear(256, self.num_actions)

    def _init_value_head(self):
        return torch.nn.Linear(256, 1)
    """
       
    
    def forward(self, inputs):
        x = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 384(16x3x8)

        """
        # Disable CNN-LSTM actor critic for now
        
        x, (hx, cx) = inputs
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 1 x 512(4x4x32)
        
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        

        value = self.value_head(x)
        return action, value, (hx, cx)       
        
        """
        probs = torch.nn.functional.softmax(self.action_head(x) /
                                             self.temperature, dim=-1)
        return probs

    # This method attach the agent to a tribe by tribal name and color
    def attach_to_tribe(self, tribe_name, tribe_color):
        self.tribe = tribe_name
        self.color = tribe_color

    # This method resets agent info 
    def reset_info(self):
        # laser parameters
        self.tagged = False        
        self.laser_fired = False
        self.US_hit = 0
        self.THEM_hit = 0
        self.in_banned = False  # 03-02-2019
        self.in_target = False
    
    # This method loads agent info 
    def load_info(self, info):
        # laser parameters
        self.tagged, self.laser_fired, self.US_hit, self.THEM_hit, self.in_banned, self.in_target = info
        
        # save in episode history (to be used in tribal reward calculation)
        self.tagged_hist.append(self.tagged)       
        self.tag_hist.append(self.laser_fired)
        self.US_hits.append(self.US_hit)
        self.THEM_hits.append(self.THEM_hit)
        self.in_banned_hist.append(self.in_banned)
        self.in_target_hist.append(self.in_target)

    # This method flush the agent's history at the end of a game episode    
    def clear_history(self):
        del self.saved_actions[:]
        del self.rewards[:]
        del self.log_probs[:]
        del self.tagged_hist[:]   
        del self.tag_hist[:]
        del self.US_hits[:]
        del self.THEM_hits[:]
        del self.in_banned_hist[:]
        del self.in_target_hist[:]

        

# Just a dumb random agent
class Rdn_Policy():
            
    def __init__(self):
        super(Rdn_Policy, self).__init__()
        
        # Tribal parameters
        self.tribe = None
        self.color = None
        
        # laser parameters
        self.tagged = False        
        self.laser_fired = False
        self.US_hit = 0
        self.THEM_hit = 0

    def select_action(self, state):
        return random.randrange(0, 8)
    
    # This method attach the agent to a tribe by tribal name and color
    def attach_to_tribe(self, tribe_name, tribe_color):
        self.tribe = tribe_name
        self.color = tribe_color
        
    # This method loads agent info 
    def load_info(self, info):
        # laser parameters
        self.tagged, self.laser_fired, self.US_hit, self.THEM_hit, self.in_banned, self.in_target = info


"""
In this file, we will implement the Tribe Class:
- Agents attached to the Tribe have a "Us" versus "Them" mentality.
- The Tribe awards agents with tribal reward based on Culture.
"""
class Tribe():
    
    color = None
    name = None
    culture = None
    members = []

    # This is run when the tribe is created
    def __init__(self, name='Vikings', color='red', culture={'name':'individualist'}, agents=[]):
        self.color = color
        self.name = name
        self.culture = culture
        self.members = agents

        # Used to implement exile and follower culture
        self.banned_zone = None
        self.target_zone = None
        
        for agent in self.members:
            agent.attach_to_tribe(name, color)
    
    def sum_rewards(self):
        # Add the rewards of the tribe members elementwise
        rewards = [0 for i in range(len(self.members[0].rewards))]
        for a in self.members:
            rewards = [rewards[j] + a.rewards[j] for j in range(len(rewards))]
        return rewards   # return the tribe's rewards 
    
    def sum_US_hits(self):
        # Add the num of friendly fires (fragging) by the tribe members elementwise
        US_hits = [0 for i in range(len(self.members[0].US_hits))]
        for a in self.members:
            US_hits = [US_hits[j] + a.US_hits[j] for j in range(len(US_hits))]
        return US_hits   # return the tribe's fragging incidents

    # 03-01-2019
    # These routines set the banned or target zones for the Tribe (only valid for certain cultures)    
    def set_banned_zone(self, banned_zone):
        # Set the zone from which agents are "banned" - receive a penalty for being there
        self.banned_zone = banned_zone

    def set_target_zone(self, target_zone):
        # Set the zone to which agents need to move towards - receive a reward for being there
        self.target_zone = target_zone

    
    def tribal_awards(self, US_hits = None, THEM_hits = None, tag_hist=None, in_banned_hist=None, in_target_hist=None):
        culture = self.culture['name']
        if culture is 'cooperative':
            coop_factor = self.culture['coop_factor']
            rewards_to_tribe = self.sum_rewards() 
            num_members = len(self.members)
            awards = [r/num_members*coop_factor for r in rewards_to_tribe]
        elif culture is 'individualist':
            rewards_to_tribe = self.sum_rewards() 
            awards = [0 for r in rewards_to_tribe]
        elif culture is 'no_fragging':
            penalty = self.culture['penalty']
            awards = [friendly_fire * penalty for friendly_fire in US_hits]
        elif culture is 'pacifist':
            penalty = self.culture['laser_penalty']
            awards = [laser * penalty for laser in tag_hist]

        # Added 03-03-2019 exile vs follower    
        elif culture is 'pacifist_exile':
            laser_penalty = self.culture['laser_penalty']
            banish_penalty = self.culture['banish_penalty']
            awards = [(laser * laser_penalty + banish * banish_penalty)  for laser, banish in zip(tag_hist, in_banned_hist)]
        elif culture is 'pacifist_follower':
            laser_penalty = self.culture['laser_penalty']
            target_reward = self.culture['target_reward']
            awards = [(laser * laser_penalty + target * target_reward)  for laser, target in zip(tag_hist, in_target_hist)]

        elif culture is 'warlike':
            penalty = self.culture['penalty']
            reward = self.culture['reward']
            awards = [friendly_fire * penalty + enemy_hit * reward \
                      for friendly_fire, enemy_hit in zip(US_hits,THEM_hits)]
        else:
            awards = None
        return awards

    
# 04-19-2019 Implemented new way to save and load checkpoints based on:
#     https://pytorch.org/tutorials/beginner/saving_loading_models.html

def save_model(file_name, ep, model, optimizer):
    """
    Save a training checkpoint whereby episode, model and optimizer parameters are saved into a dictionary 
    prior to being saved into the model file.
    
    Refer to:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """

    torch.save({
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
    }, file_name) 
    
    return
    
def load_model(agent, optimizer, model_file):
    """
    Convert a model file saved using pickle into a file where model, optimizer and epoch parameters
    are saved into a dictionary prior being saved.
    
    Refer to:
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
    
    The original file is assumed to be under orig_dir. The converted file is saved under conv_dir.
    """
    checkpoint = torch.load(model_file)
    episode = checkpoint['epoch']
    agent.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return episode


def finish_episode(tribes, learners, optimizers, gamma, cuda):
    """ 
    In RL, policy gradient is calculated at the end of an episode and only then used to update
    the weights of an agent's policy.
    
    The code perform policy update on each learning agent independently. Reward for each time 
    step is stored in the list policy.rewards[] --> r(t)
    """  
    
    num_learners = len(learners)
    total_norms = [0 for i in range(num_learners)]
    policy_losses = [[] for i in range(num_learners)]
    losses = [[] for i in range(num_learners)]
    T_reward = []

   
    for i in range(num_learners):

        R = 0
        saved_actions = learners[i].saved_actions
        
        for t in tribes:
            if t.name is learners[i].tribe:
                
                # Based on team culture, calculate the team reward for the agent    
                culture = t.culture['name']
            
                if culture is 'cooperative':
                    T_reward = t.tribal_awards()
                elif culture is 'individualist':
                    T_reward = t.tribal_awards()
                elif culture is 'no_fragging':
                    T_reward = t.tribal_awards(US_hits = learners[i].US_hits)
                elif culture is 'pacifist':
                    T_reward = t.tribal_awards(tag_hist = learners[i].tag_hist)
                elif culture is 'pacifist_exile':
                    T_reward = t.tribal_awards(tag_hist = learners[i].tag_hist, \
                                           in_banned_hist=learners[i].in_banned_hist)
                elif culture is 'pacifist_follower':
                    T_reward = t.tribal_awards(tag_hist = learners[i].tag_hist, \
                                           in_target_hist=learners[i].in_target_hist)
                elif culture is 'warlike':
                    T_reward = t.tribal_awards(US_hits = learners[i].US_hits,THEM_hits = learners[i].THEM_hits)
                else:
                    T_reward = t.tribal_awards()
 
                # For debug only
                # print('Agent{} receives tribal award from Tribe{}'.format(i,t.name))
                # print (T_reward)
                # print (learners[i].rewards)
                
        # Do not implement actor-critic for now
        # value_losses = []
        
        rewards = deque()

        for r,T in zip(learners[i].rewards[::-1],T_reward[::-1]):
            # The agent is incentivized to cooperate by an award of 30% of what the tribe takes
            # in by all its members
            R = r + T + gamma * R
            rewards.appendleft(R)
            
        rewards = list(rewards)
        rewards = torch.Tensor(rewards)
        if cuda:
            rewards = rewards.cuda()

        # z-score rewards
        rewards = (rewards - rewards.mean()) / (1.1e-7+rewards.std())
        
        #Debug     
        #print (rewards)       
        
        """
        Do not implement actor-critic for now!!!
        for (log_prob, state_value), r in zip(saved_actions, rewards):
            reward = r - state_value.data[0]
            policy_losses.append(-log_prob * Variable(reward))
            r = torch.Tensor([r])
            if cuda:
                r = r.cuda()
            value_losses.append(torch.nn.functional.smooth_l1_loss(state_value,
                                                               Variable(r)))

        optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()        
        
        
        """
        for log_prob, r in zip(saved_actions, rewards):
            r = torch.Tensor([r])
            if cuda:
                r = r.cuda()
            policy_losses[i].append(-log_prob * Variable(r))

        optimizers[i].zero_grad()
        losses[i] = torch.stack(policy_losses[i]).sum()
        losses[i].backward()
        
        # Gradient Clipping Update: prevent exploding gradient
        total_norms[i] = torch.nn.utils.clip_grad_norm_(learners[i].parameters(), 8000)
        
        optimizers[i].step()
        learners[i].clear_history()   # clear an agent's history at the end of episode


    return total_norms