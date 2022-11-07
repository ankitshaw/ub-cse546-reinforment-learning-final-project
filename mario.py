import gym
from gym import spaces

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



STATES = [['MS' , '  ' , '  ' , '  ' , '  ' ],
          ['  ' , '  ' , '  ' , '  ' , 'FB' ],
          ['FF' , 'FF' , '  ' , 'CR' , '  ' ],
          ['  ' , '  ' , 'FB' , '  ' , '  ' ],
          ['FT' , '  ' , '  ' , '  ' , 'SR' ]]

ACTIONS = {
        0: ['L',(0,-1)], #Left
        1: ['U',(-1,0)], #Up
        2: ['R', (0,1)],  #Right
        3: ['D',(1,0)]   #Down
    }
REWARDS = {
        'FB' : -2, #Fire Ball
        'FF' : -3, #Fire Flower
        'CR' : 2,  #Coin Reward
        'SR' : 4,  #Star Reward
        'FT' : 10  #Flag Terminal State
    }

# Icons to be used for Visualization
ICONS = {
        '-2' : "./images/fire_ball.png",
        '-3' : "./images/fire_flower.png",
        '2'  : "./images/coin_reward.png",
        '4'  : "./images/star_reward.png",
        '10' : "./images/flag_goal.png",
        "MS" : "./images/mario_agent.png",
        "-3MS" : "./images/fire_flower_mario.png",
        "-2MS" : "./images/fire_ball_mario.png",
        "2MS"  : "./images/coin_reward_mario.png",
        "4MS"  : "./images/star_reward_mario.png",
        "10MS" : "./images/flag_goal_mario.png"

    }

class MyMarioEnvironment(gym.Env):
    def __init__(self, environment:list[list]=STATES, actions:dict=ACTIONS, rewards:dict=REWARDS, p_transition:float=1.0, environment_type:str='deterministic'):
        """This function is used to initialize the Environment Class

        Args:
            environment (list[list], optional): Represents the environment defined with characters. Defaults to STATES.
            actions (dict, optional): Map of actions allowed in the environment. Defaults to ACTIONS.
            rewards (dict, optional): Map of rewards allowed in the environment. Defaults to REWARDS.
            p_transition (float, optional): Probability of transition used for Stochastic Environment. Defaults to None. If value is 1 then acts as Deterministic environment
            epsilon (float, optional): Used to exploration and exploitation policy (Not used for Part1). Defaults to None.
            discount_factor (float, optional): Used to discount the rewards (Not used for Part1). Defaults to None.
            environment_type (str, optional): The type of environment. Takes input 'deterministic' or 'stochastic'. Defaults to None.
        """

        self.environment = environment
        self.env_row, self.env_col = len(self.environment), len(self.environment[0])
        self.states, self.start_pt, self.end_pt, self.current_pt, self.current_state = self._get_state_space(environment, rewards)
        self.observation_space = spaces.Discrete(self.env_row*self.env_col)
        self.action_space = spaces.Discrete(len(actions))
        self.environment_type = environment_type
        self.p_transition = p_transition
        self.observation = self.states.flatten()
        self.rewards = rewards
        self.rewards_space = self._get_reward_space(environment, rewards)
        self.render_help =  self._get_reward_space(environment, rewards)
        self.actions = actions
        self.action_letters = self._get_action_letter_map(actions)
        self.T_M = self._get_transition_matrix()
        self.current_action = -1
        self.current_action_index = -1
        self.reward_states_gained = []

    def step(self, action:str):
        """Step function that perform next action. 

        Args:
            action (str): the action taken at a time step

        Returns:
            observation, reward, done, info: return the observation after action, the reward received, whether max timestep reached and other debug info
        """
        done = False
        info = {}
        if action not in self.action_letters.keys():
            self.new_reward = 0
        else:
            self.current_action = self.action_letters[action]
            t_action = self.T_M[self.current_state][self.current_action]
            t_prob = [t[0] for t in t_action]
            self.current_action_index = np.random.choice(a = len(t_action), p = t_prob)
            prob, new_pt, new_reward, done = t_action[self.current_action_index]
            self.previous_state = self.current_state
            self.current_state = self._get_state_from_xy(*new_pt)
            self.current_pt = new_pt
            self.state = self._get_state_from_xy(*self.current_pt)
            if self.state in self.reward_states_gained:
                self.new_reward = 0
            else:
                self.new_reward = new_reward
                if new_reward > 0:
                    self.reward_states_gained.append(self.state)
            self.states[self.current_pt[0]][self.current_pt[1]] = -5
            self.observation = self.states.flatten()
            self.reward_states_gained = [] if done else self.reward_states_gained
            info = {'action_performed':self.actions[self.current_action_index][0], 'prob':prob}
        return self.observation, self.new_reward, done, info
        
    def reset(self):
        """Reset's the environment with initial values

        Returns:
            observation: return the observation after reset
        """
        self.timestep = 0
        self.states, self.start_pt, self.end_pt, self.current_pt, self.current_state = self._get_state_space(self.environment, self.rewards)
        self.state = self._get_state_from_xy(*self.current_pt)
        self.observation = self.states.flatten()
        return self.observation

    def render(self, mode:str="rgb", icons:dict=ICONS):
        """Renders the environment

        Args:
            mode (str, optional): Mode of rendering whether RGB or pictorial. Defaults to "rgb".
            icons (dict, optional): Image locations for pictorial mode. Defaults to ICONS.
        """
        if mode == "rgb":
            plt.imshow(self.states) 
            plt.show()
        elif mode == "human":
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 5)
            im = plt.imread("./images/bg.jpg")
            im = ax.imshow(im)
            for r in range(self.env_row):
                for c in range(self.env_col):
                    if np.array_equal(self.current_pt,[r,c]) and np.array_equal(self.render_help[r][c],0):
                        img = icons['MS']
                    elif np.array_equal(self.current_pt,[r,c]) and not np.array_equal(self.render_help[r][c],0):
                        img = icons[str(int(self.render_help[r][c]))+"MS"]
                        if self.state in self.reward_states_gained:
                            self.rewards_space[r][c] = 0
                    elif not np.array_equal(self.current_pt,[r,c]) and not np.array_equal(self.rewards_space[r][c],0):
                        img = icons[str(int(self.rewards_space[r][c]))]
                    else:
                        continue

                    agent = AnnotationBbox(OffsetImage(plt.imread(img), zoom=0.5), np.add((c,4-r), [0.5, 0.5]), frameon=False)
                    ax.add_artist(agent)      

            plt.xticks([0, 1, 2, 3, 4])
            plt.yticks([0, 1, 2, 3, 4])
            plt.grid()  
            plt.show()

    def _get_action_letter_map(self, actions):
        action_letter = {}
        for a,v in actions.items():
            action_letter[v[0]] = a
        return action_letter

    def _get_state_space(self, environment:list[list]=None, rewards:dict=None):
        state_space = np.zeros((self.env_row,self.env_col))
        for r in range(self.env_row):
            for c in range(self.env_col):
                if environment[r][c] in rewards:
                    state_space[r][c] = rewards[environment[r][c]]
                
                if environment[r][c] == 'MS':
                    state_space[r][c] = -5 #if states is used for rendering
                    start = np.array((r,c))
                elif environment[r][c] == 'FT':
                    end = np.array((r,c))
                
        return state_space, start, end, start, self._get_state_from_xy(*start)
    
    def _get_reward_space(self, environment:list[list]=None, rewards:dict=None):
        rewards_space = np.zeros((self.env_row,self.env_col))
        for r in range(self.env_row):
            for c in range(self.env_col):
                if environment[r][c] in rewards:
                    rewards_space[r][c] = rewards[environment[r][c]]                
        return rewards_space

    def _get_reward(self, state_pt):
        self.new_reward = self.rewards_space[state_pt[0]][state_pt[1]]
        return self.new_reward
    
    def _update_reward(self, action, action_index):
        if self.T_M[self.previous_state][action][action_index][2] > 0:
            self.rewards_space[self.current_pt[0]][self.current_pt[1]] = 0
            self.T_M[self.previous_state][action][action_index][2] = 0

    def _get_next_pt(self, action, pt=None):
        if pt is None:
            pt = self.current_pt
        new_pt = pt + action
        if new_pt[0] < 0 or new_pt[0] == self.env_row or new_pt[1] < 0 or new_pt[1] == self.env_col:
            return pt
        else:
            return new_pt

    def _get_state_from_xy(self, row, col):
        return row*self.env_col + col
    
    def _get_step_transition(self, current_pt, current_action):
        next_pt = self._get_next_pt(action=self.actions[current_action][1], pt = current_pt)
        new_state = self._get_state_from_xy(*next_pt)
        reward = self._get_reward(next_pt)
        done = True if np.array_equal(self.end_pt,next_pt) else False
        return next_pt, reward, done 
    
    def _get_transition_matrix(self):
        T_M = {s: {a: [[] for a in range(self.action_space.n)] for a in range(self.action_space.n)} for s in range(self.observation_space.n)}
        for row in range(self.env_row):
            for col in range(self.env_col):
                s = self._get_state_from_xy(row,col)
                for a in range(self.action_space.n):
                    li = T_M[s][a]
                    if self.environment_type=='stochastic':
                        li[a] = [self.p_transition, *self._get_step_transition(np.array((row,col)),a)]
                        p_tran_others = (1 - self.p_transition)/3
                    else:
                        li[a] = [1.0, *self._get_step_transition(np.array((row,col)),a)]
                        p_tran_others = 0.0
                    
                    for ac in range(self.action_space.n):
                            if ac != a:
                                li[ac] = [p_tran_others, *self._get_step_transition(np.array((row,col)),ac)]
        return T_M