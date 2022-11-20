import gym
from gym import spaces
import numpy as np

""" Gym Environment Implementation """

class Sokoban(gym.Env):
  # constructor function
  def __init__(self, width, height):
      self.width = width                    # width of maze
      self.height = height                  # height of maze
      self.state = np.array(np.zeros((self.width,self.height), dtype=int))  # whole grid representation
      self.r_step = -1                      # reward for moving
      self.r_wait = -1                      # reward for waiting while agent finds box
      self.r_box = 10                       # reward for moving box to target position   
      self.r_game_over = -10                # reward for moving box to dead end so game over
      
      self.action_space = spaces.Discrete(4)  # 4 actions of boxes: UP, DOWN, LEFT, RIGHT
      
      self.box_location = self._generate_box()              # box location intialized
      self.agent_location = self._generate_agent_position() # agent location intialized
     
      # Actions
      # =================================
      # +ve x => [+1,0], +ve y => [0, +1]
      self.direction = {
        0: [-1, 0], # UP
        1: [0, 1],  # RIGHT
        2: [1, 0],  # DOWN
        3: [0, -1], # LEFT 
      }

      self.bound = self.width-1   # to check for boundary for agent
      self.done = False           # done status for game to end or move on
      self.wincount = 0           # win counter to output number of wins

  # def _game_walls(self):
  #   for i in range(len(self.state)):
  #     for j in range((len(self.state))):
  #       if (i == 0 or i == (len(self.state)-1)):
  #         self.state[i][j] = 1
  #       elif (j == 0 or j == (len(self.state)-1)):
  #         self.state[i][j] = 1
  #   return self.state

  # def _generate_random_walls(self):
  #   for i in range(len(self.state)-1):
  #     for j in range(len(self.state)-1):
  #       if (i == 1 or i == (len(self.state)-2)):
  #         random_j = np.random.randint((len(self.state)))
  #         self.state[i][random_j] = 1
  #       elif (j == 1 or j == (len(self.state)-2)):
  #         random_i = np.random.randint((len(self.state)))
  #         self.state[random_i][j] = 1
  #   return self.state


  # We keep the boxes and target position fixed and hard code it for now
  # Walls are imaginary

  # box symbol: 2
  # box target symbol: 100
  # agent symbol: 1

  # Generating box position
  def _generate_box(self):
    self.state[1][3] = 2
    return [1,3]

  # Generating box target position
  def _generate_box_target(self):
    self.state[2][2] = 100
    return [2,2]

  # Agent position is hard coded for simplicity but can be spawned randomly
  def _generate_agent_position(self):
    # x = np.random.randint(10)
    # y = np.random.randint(10)
    # if ([x,y] == self._generate_box()).all() or ([x,y] == self._generate_box_target()).all():
    #   x = np.random.randint(10)
    #   y = np.random.randint(10)
    self.state[0][0] = 1
    return [0,0]

  # Step function for each action returns reward, next state and done status
  def step(self, action):
    box_target = self._generate_box_target()
    box_loc = self.box_location

    # new agent_location after action 
    new_agent_loc = np.add([self.agent_location[0], self.agent_location[1]], self.direction[action]) 

    # UP
    if action == 0 and self.agent_location[0] >= 0 and self.agent_location[1] >= 0:
      if (new_agent_loc == box_loc).all(): return self._box_agent_move(action, new_agent_loc, box_loc, box_target)
      else: return self._agent_move(action, new_agent_loc, box_loc, box_target)

    # RIGHT    
    elif action == 1 and self.agent_location[0] >= 0 and self.agent_location[1] >= 0 and self.agent_location[0] <= self.bound:
      if (new_agent_loc == box_loc).all(): return self._box_agent_move(action, new_agent_loc, box_loc, box_target)
      else: return self._agent_move(action, new_agent_loc, box_loc, box_target)

    # DOWN
    elif action == 2 and self.agent_location[0] <= self.bound and self.agent_location[1] <= self.bound:
      if (new_agent_loc == box_loc).all(): return self._box_agent_move(action, new_agent_loc, box_loc, box_target)
      else: return self._agent_move(action, new_agent_loc, box_loc, box_target)

    # LEFT
    elif action == 3 and self.agent_location[0] >= 0 and self.agent_location[1] >= 0 and self.agent_location[0] <= self.bound and self.agent_location[1] <= self.bound:
      if (new_agent_loc == box_loc).all(): return self._box_agent_move(action, new_agent_loc, box_loc, box_target)
      else: return self._agent_move(action, new_agent_loc, box_loc, box_target)

    # edge case condition 
    else:
      print("Not going anywhere")
      return None


  # ONLY called when agent finds a box
  def _box_agent_move(self, action, new_agent_loc, box_loc, box_target):
    new_box_loc = np.add(box_loc, self.direction[action])
    # print("BOX DETECTED !!")

    # if we detect the box and move it, we check for all the conditions for box.
    # ==================================================================================
    # if new box_location matches the target box_location, then game is over and you WIN
    if (new_box_loc == box_target).all():
      self.done = True                                    # update done stats to True
      self.box_location = new_box_loc
      print("***** You Win *****")
      self.wincount += 1                                  # increase win counter by 1 if won
      # box update in state
      self.state[new_box_loc[0]][new_box_loc[1]] = 2      # updating state with new box_location
      self.state[box_loc[0]][box_loc[1]] = 0              # updating old box_location to 0
      # agent pos update in state
      self.state[new_agent_loc[0]][new_agent_loc[1]] = 1
      self.state[self.agent_location[0]][self.agent_location[1]] = 0
      # self.reset()
      self.agent_location = new_agent_loc
      self.box_location = new_box_loc
      # print(self.state)
      return new_agent_loc, self.r_box, self.done, {}

    # if new box_location is on boundary ==> GAME OVER, game resets
    elif (((new_box_loc == np.array([self.width-1,self.height-1])).any() or (new_box_loc == np.array([0,0])).any() or (new_box_loc == np.array([0,0])).any() or (new_box_loc == np.array([0,0])).any())): # and not (((box_target == np.array([self.width-1,self.height-1])).any() or (box_target == np.array([0,0])).any())):
      self.done = True
      # print("Box on boundary")
      
      # box update in state
      self.state[new_box_loc[0]][new_box_loc[1]] = 2      # updating state with new box_location
      self.state[box_loc[0]][box_loc[1]] = 0              # updating old box_location to 0
      # agent pos update in state
      self.state[new_agent_loc[0]][new_agent_loc[1]] = 1
      self.state[self.agent_location[0]][self.agent_location[1]] = 0
      self.agent_location = new_agent_loc
      self.box_location = new_box_loc
      # print(self.state)
      return new_agent_loc, self.r_game_over, self.done, {}

    # checking corner conditions of sokoban game, eg: [0,0], [10,10], [0,10], [10,0]
    # elif ((new_box_loc == np.array([self.width-1,self.height-1])).all()) or ((new_box_loc == np.array([0,0])).all()) or ((new_box_loc == np.array([0,self.height-1])).all()) or ((new_box_loc == np.array([self.width-1,0])).all()):
    #   self.done = True
      
    #   print("Box in corners")
    #   # print("You Lose")
    #   # self.reset()
    #   # box update in state
    #   self.state[new_box_loc[0]][new_box_loc[1]] = 2      # updating state with new box_location
    #   self.state[box_loc[0]][box_loc[1]] = 0              # updating old box_location to 0
    #   # agent pos update in state
    #   self.state[new_agent_loc[0]][new_agent_loc[1]] = 1
    #   self.state[self.agent_location[0]][self.agent_location[1]] = 0
    #   self.agent_location = new_agent_loc
    #   self.box_location = new_box_loc
    #   print(self.state)
    #   return new_agent_loc, self.r_game_over, self.done, {}

    # elif ((box_target == np.array([0,0])).any() or (box_target == np.array([self.width-1,self.height-1])).any()) and (new_box_loc[0] == box_target[0]) or (new_box_loc[1] == box_target[1]):
    #   print("Box Moving on boundary")
    #   # box update in state
    #   self.state[new_box_loc[0]][new_box_loc[1]] = 2      # updating state with new box_location
    #   self.state[box_loc[0]][box_loc[1]] = 0              # updating old box_location to 0
    #   # agent pos update in state
    #   self.state[new_agent_loc[0]][new_agent_loc[1]] = 1
    #   self.state[self.agent_location[0]][self.agent_location[1]] = 0
    #   # old values updated to new
    #   self.agent_location = new_agent_loc
    #   self.box_location = new_box_loc
    #   return new_agent_loc, self.r_step, self.done, {}

    # else we update old box_location and go to next step
    else:
      # box update in state
      self.state[new_box_loc[0]][new_box_loc[1]] = 2      # updating state with new box_location
      self.state[box_loc[0]][box_loc[1]] = 0              # updating old box_location to 0
      # agent pos update in state
      self.state[new_agent_loc[0]][new_agent_loc[1]] = 1
      self.state[self.agent_location[0]][self.agent_location[1]] = 0
      # old values updated to new
      self.agent_location = new_agent_loc
      self.box_location = new_box_loc
      return new_agent_loc, self.r_step, self.done, {}
    
  # ONLY called when agent is trying to find box  
  def _agent_move(self, action, new_agent_loc, box_loc, box_target):
    new_box_loc = box_loc
    # self._print_agent_action(action)

    if new_agent_loc[0] == -1 or new_agent_loc[1] == -1 or new_agent_loc[0] > self.bound or new_agent_loc[1] > self.bound:
      # print("Invalid move")
      # new_agent_loc = self.agent_location
      return self.agent_location, self.r_wait, self.done, {} # return (next state, reward=0, done=False, _)
  
    # if move is valid agent moves and gets 0 reward
    else:   
      # print("Valid move")
      self.state[new_agent_loc[0]][new_agent_loc[1]] = 1
      if (np.array(self.agent_location) == box_target).all():
        self.state[self.agent_location[0]][self.agent_location[1]] = 100
      else:
        self.state[self.agent_location[0]][self.agent_location[1]] = 0
      self.agent_location = new_agent_loc
      return new_agent_loc, self.r_wait, self.done, {}

  # reset the grid to initial state of box and box target. Also reset last box_location of box before game over or win
  def reset(self):
    # print("********** GAME RESET ***********\n")
    self.state = np.array(np.zeros((self.width,self.height), dtype=int))
    self.box_location = self._generate_box()
    self._generate_box_target()
    self.agent_location = self._generate_agent_position()
    self.done = False
    return self.agent_location

  # print function for each action of box
  def _print_agent_action(self, action):
    if action == 0:
      print("UP")
    elif action == 1:
      print("RIGHT")
    elif action == 2:
      print("DOWN")
    elif action == 3:
      print("LEFT")

  def _print_box_agent_move(self, action):
    if action == 0:
      print("BOX PUSHED UP")
    elif action == 1:
      print("BOX PUSHED RIGHT")
    elif action == 2:
      print("BOX PUSHED DOWN")
    elif action == 3:
      print("BOX PUSHED LEFT")



""" Creating an instance of Sokoban Game class """
env = Sokoban(6,6)

"""======================================================================================="""

""" TD (0): SARSA Implementation"""

# Epsilon greedy for SARSA
def eps_greedy(env, eps, q_values):
  actions = list(env.direction.keys())
  if np.random.random() > eps:
    # print("exploiting")
    # print("q_values for the state:", q_values)
    return actions[np.argmax(q_values)]
  else:
    # print("exploring")
    return np.random.choice(actions)

# SARSA algorithm implementation
def SARSA(env, alpha, gamma, E):
  # qsa matrix stores array of 4 action values / expected return for every state
  q_sa = np.full((env.width, env.height, len(env.direction)), fill_value=np.float(-100))
  steps_list = []     # To count steps for each episode
  q_values = {}       # To store q_values for all initial state for each episode

  # Looping for each episode in E
  for episode in range(E):
    print(f"Episode: {episode}")
    # print("*******************\n")
    s = env.reset()     # we reset initial stae for each episode
    done = False        # we initialize done status for while loop to False
    a = eps_greedy(env, 0.6, q_sa[s[0]][s[1]])  # initialize action using e-greedy action selection
    steps = 0           # step counter

    # while agent has not reached terminal state or game is not over
    while done == False:
      steps += 1
      # print(f"Episode -> {episode} : step: {steps}")
      s_p, r, done, _ = env.step(a)           # step function returning next state, reward and done satus
      a_p = eps_greedy(env, 0.6, q_sa[s_p[0]][s_p[1]])  # next action selection using e-greedy method

      # q_sa update function
      q_sa[s[0]][s[1]][a] += alpha*(r + gamma*q_sa[s_p[0]][s_p[1]][a_p] - q_sa[s[0]][s[1]][a])
      
      # update previous state to new state and previous action to new action
      s = s_p
      a = a_p

      # After updating qsa-function, if next state is terminal state and done status is True,
      # We end episode and reset game
      if env.done == True:
        done = True
        steps_list.append(steps)
        
    # making dictionary of qsa-values for initial value    
    q_values[episode] = np.max(q_sa[0][0])
    # generatePolicy(env, q_sa)

  return q_sa, q_values, steps_list

# generating policy using qsa matrix
def generatePolicy(env, qsa):
  policy = np.zeros((env.width, env.height), dtype=str)
  v = np.zeros((env.width, env.height))
  action_symbols = {0: "U", 1: "R", 2: "D", 3: "L"}

  for i in range(env.width):
    for j in range(env.height):
      actions = qsa[i][j]
      a = np.argmax(actions)
      value = np.max(actions)
      policy[i][j] = action_symbols[a]
      v[i][j] = value

  return policy, v

"""================================================================================="""

""" Semi-Gradient SARSA Implementation"""

""" 
This code has been adapted from my assignment-2 code and has 
been modified for Sokoban Game to make comparison with my
SARSA algorithm for plotting results and is not my primary 
objective of project. 
"""
# epsilon greedy algorithm
def semigradient_eps_greedy(env, s, w, eps):
    if np.random.random() > eps:    # if random value > eps => Exploit the knowledge
      q_list = []                   # create list of q-values
      for a in range(len(env.direction.keys())):  # for each action calculate q_hat value
        q_value = q_hat(s, a, w)
        q_list.append(q_value)
      if sum(q_list) == 0:            # check if list has all q_hat values 0
        a = np.random.choice(q_list)  # then action will be chosen randomly to avoid choosing first action every time 
      else:
        a = np.argmax(q_list)         # else we select action with highest q_hat value
      return a
      
    else:
      a = np.random.choice(list(env.direction.keys())) # choose action randomly => Exploring
      return a

# q-hat value
def q_hat(s, a, w):
    q_hat = np.dot(w, x_sa(s, a))  # q_hat(s,a,w) = w x transpose(x_sa)
    return q_hat

# Feature Vector
"""
The feature vector has been inspired from classroom notes and from 
Section-9.5: Feature Construction for Linear Methods by Sutton and Barto

Feature vector of form: [1, x, y, x_square, y_square, xy, x+a[0], y+a[1]] 
"""
def x_sa(s, a):
    x = np.zeros((8, ))
    x[0] = 1
    x[1] = s[0]
    x[2] = s[1]
    x[3] = s[0]**2
    x[4] = s[1]**2
    x[5] = s[0]*s[1]
    x[6] = s[0] + env.direction[a][0]
    x[7] = s[1] + env.direction[a][1]
    return x

def SemiGradientSARSA(env, eps, alpha, E):
    w = np.full((8,), fill_value=float(-100))      # initialize weight vector
    # w = np.zeros((8,))
    gamma = 0.9               # gamma value
    episode = 0               # initialize episode count
    steps_dict = {}           # dictionary to store number of steps taken for each episode
    weights_dict = {}         # dictionary to store all weights for each episode
    initial_state_w = {}

    for _ in range(E):        # for each episode
        print(f"Episode:{episode}")

        steps = 0
        s = env.reset()       # generate random agent position for each episode
        a = semigradient_eps_greedy(env, s, w, eps) # selecting action using e-greedy algorithm
        
        done = False          # done for SARSA algorithm initiated to False
        while done==False:    # Until terminal state is reached, go over the loop
            steps += 1
            s_p, r, done, _ = env.step(a)  # calling step function for action selected
        
            # if terminal state is reached agent WINS or LOSES
            if done==True:      
                w += alpha*(r - q_hat(s, a, w))*x_sa(s, a)  # update final weight value

            # else we update weight vector until we reach terminal state
            else:
                a_p = semigradient_eps_greedy(env, s_p, w, eps)   # selecting next action 
                w += alpha*(r + gamma*q_hat(s_p, a_p, w) - q_hat(s, a, w))*x_sa(s, a) # updating weight vector
      
                s = s_p       # updating current state with next state for next step
                a = a_p       # updating current action with next action for next step

        done = False                # again update done=True to done=False to start new episode
        steps_dict[episode] = steps # storing number of steps for each episode in dictionary
      
        # Only for plotting evolution of value function for initial state [0][0]
        temp_list = []
        for i in range(4):
            q_hat_initial_state = np.dot(w, x_sa([0,0], i))
            temp_list.append(q_hat_initial_state)

        initial_state_w[episode] = np.max(temp_list)      # storing values for initial state for each episode
        weights_dict[episode] = w.copy()                  # store final weight vector for the episode in dictionary
        episode += 1

    return weights_dict, steps_dict, initial_state_w


def optimalPolicy(env, weight):
    q_values = np.full((env.width,env.height, 4), fill_value=np.float(-100))
    policy = np.zeros((env.width,env.height), dtype=str)    # policy matrix of size as maze
    values = np.zeros((env.width,env.height))               # value matrix of size as maze
    actions = list(env.direction.keys())           # list of action values
    a_dict = {0:"U", 1:"R", 2:"D", 3:"L"}           # dictionary of each action with their symbol for better policy representation

    for i in range(env.width):
      for j in range(env.height):
        q_value_list = []
        s = np.array([i,j])                         # each state in matrix

        for action in actions:                      # for each action, calculate q-hat value
          q_value = q_hat(s, action, weight)
          q_values[i][j][action] = q_value
          q_value_list.append(q_value)
        
        temp_max_value = np.max(q_value_list)       # find maximum q-hat value out of 4 actions
        temp_a = np.argmax(q_value_list)            

        values[i][j] = temp_max_value               # assign action value to each state in value matrix
        policy[i][j] = a_dict[temp_a]               # assign action symbol to each state in matrix
      
    return values.round(3), policy, q_values  


if __name__ == "__main__":

  optimal_qsa, qsa_initial_state, steps = SARSA(env, 0.01, 0.9, 1000)
  # weights, steps, init_state = SemiGradientSARSA(env, 0.6, 0.0001, 5000)
  

  episodes = list(qsa_initial_state.keys())
  # print("Win Percent:", env.wincount/(episodes[-1]+1)*100)
  # episodes = [10, 100, 500, 1000, 5000]
  # wins = []
  # for episode in episodes:
  #   SemiGradientSARSA(env, 0.6, 0.001, episode)
  #   wins.append(env.wincount/(episode)*100)
  # print(wins)
    
  p_matrix, q_matrix = generatePolicy(env, optimal_qsa)
  # value_matrix, policy_matrix, q_matrix = optimalPolicy(env, weight=weights[episodes[-1]-1])
  print("\n")
  print("Policy matrix SARSA:\n",p_matrix,"\n")
  print("Value matrix SARSA:\n",q_matrix)
  
  # plt.plot(episodes, list(qsa_initial_state.values()), color="r", label="SARSA alpha=0.1")
  # # plt.plot(episodes, list(init_state.values()), color="b", label="Semi-Gradient SARSA alpha=0.0001")

  # plt.xlabel("Episodes")
  # plt.ylabel("Expected Return")
  # plt.legend()
  # plt.title(f"Policy Evolution over {episodes[-1]+1} episodes")

  # plt.show()