  # Gym environment
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

""" Introduction """
""" ============ """

"""
For running the program,
It will ask for all the input values for successfully running the
program. 

** Only enter 1 value for each input **

Error:

There is an error in the final result of policy matrix which only
gives 1 single action for all states. I tried figuring out but 
was unsuccessful. Though, according to my understanding value matrix
works fine.

"""


""" Gym Environment """
""" =============== """

# Gym environment class MazeEscape
class MazeEscape(gym.Env):
  def __init__(self, width, height):
    self.maze_width = width         # width of maze
    self.maze_height = height       # height of maze
    self.maze = np.zeros((self.maze_width, self.maze_height), dtype=int) # visualization of maze

    self.r_exit = 10                # reward for finidng the exit
    self.r_move = -1                # reward for moving in any direction
    self.r_deadmove = -1            # reward for moving in direction of wall-dead move

    self.done = False               # done status of environment

    self.action_space = spaces.Discrete(4)  # 4 discrete values for actions 
    self.direction = {              # 4 actions with their values
        0: [-1, 0], # UP
        1: [0, 1],  # RIGHT
        2: [1, 0],  # DOWN
        3: [0, -1]  # LEFT
    }

    self.maze[self.maze_width - 1][self.maze_height - 1] = 100    # setting target location as 100  
    self.exit = np.array([self.maze_width-1, self.maze_height-1]) # exit at end corners
    self.agent_pos = np.array(self._generate_agent_location())    # initializing agent position randomly 

  # Generating random agent location
  def _generate_agent_location(self):
    self.random_x = np.random.randint(self.maze_width)
    self.random_y = np.random.randint(self.maze_height)
    pos = [self.random_x, self.random_y]

    if (pos != self.exit).all():      # checking if randomly generated location is not target location
      self.maze[pos[0]][pos[1]] = 1
      return pos
    else:
      pos = [0,0]                             # if randomly generated location is target location, we initialize as [0,0]
      self.maze[pos[0]][pos[1]] = 1
      return pos

  # Step function following environment guidelines
  def step(self, action):
    
    # if condition for every action
    if action == 0:
      next_state = np.add(self.agent_pos,self.direction[0])   # calculating next state using agent position and action 
      
      # checking if next state is target state
      if np.array_equal(next_state, self.exit):       
        # print("WIN")
        self.done = True                              # update done to True
        # self.reset()
        return next_state, self.r_exit, self.done, {} # return (next state, reward=10, done=True, _)

      # if move is invalid, agent remains in same position and move to next step
      elif next_state[0] == -1 or next_state[1] == -1 or next_state[0] >= self.maze_width or next_state[1] >= self.maze_height:
        # print("Invalid move")
        return self.agent_pos, self.r_deadmove, self.done, {} # return (next state, reward=-1, done=False, _)
      
      # if move is valid agent moves and gets -1 reward
      else:   
        # print("MOVED UP")
        self.maze[next_state[0]][next_state[1]] = 1
        self.maze[self.agent_pos[0]][self.agent_pos[1]] = 0
        self.agent_pos = next_state
        return self.agent_pos, self.r_move, self.done, {}

    elif action == 1:
      next_state = np.add(self.agent_pos,self.direction[1])
      if np.array_equal(next_state,self.exit):
        # print("WIN")
        self.done = True
        # self.reset()
        return next_state, self.r_exit, self.done, {}
      elif next_state[0] == -1 or next_state[1] == -1 or next_state[0] >= self.maze_width or next_state[1] >= self.maze_height:
        # print("Invalid move")
        return self.agent_pos, self.r_deadmove, self.done, {}
      else:
        # print("MOVED RIGHT")
        self.maze[next_state[0]][next_state[1]] = 1
        self.maze[self.agent_pos[0]][self.agent_pos[1]] = 0
        self.agent_pos = next_state
        return self.agent_pos, self.r_move, self.done, {}
    
    elif action == 2:
      next_state = np.add(self.agent_pos, self.direction[2])
      if np.array_equal(next_state, self.exit):
        # print("WIN")
        self.done = True
        # self.reset()
        return next_state, self.r_exit, self.done, {}
      elif next_state[0] == -1 or next_state[1] == -1 or next_state[0] >= self.maze_width or next_state[1] >= self.maze_height:
        # print("Invalid move")
        return self.agent_pos, self.r_deadmove, self.done, {}
      else:
        # print("MOVED DOWN")
        self.maze[next_state[0]][next_state[1]] = 1
        self.maze[self.agent_pos[0]][self.agent_pos[1]] = 0
        self.agent_pos = next_state
        return self.agent_pos, self.r_move, self.done, {}

    elif action == 3:
      next_state = np.add(self.agent_pos, self.direction[3])
      if np.array_equal(next_state, self.exit):
        # print("WIN")
        self.done = True
        # self.reset()
        return next_state, self.r_exit, self.done, {}
      elif next_state[0] == -1 or next_state[1] == -1 or next_state[0] >= self.maze_width or next_state[1] >= self.maze_height:
        # print("Invalid move")
        return self.agent_pos, self.r_deadmove, self.done, {}
      else:
        # print("MOVED LEFT")
        self.maze[next_state[0]][next_state[1]] = 1
        self.maze[self.agent_pos[0]][self.agent_pos[1]] = 0
        self.agent_pos = next_state
        return self.agent_pos, self.r_move, self.done, {}
    
  # reset function to reset the state, maze, agent position and done status
  def reset(self):
    self.maze = np.zeros((self.maze_width, self.maze_height), dtype=int)
    self.agent_pos = self._generate_agent_location()
    self.maze[self.agent_pos[0]][[self.agent_pos[0]]] = 1
    self.maze[self.maze_width - 1][self.maze_height - 1] = 100
    self.done = False
    return self.agent_pos


""" Semi gradient SARSA implementation """
""" ================================== """

if __name__ == "__main__":

  WIDTH = int(input("Enter desired maze width: "))
  HEIGHT = int(input("Enter desired maze height: "))
  EPISODES = int(input("Enter number of episodes: "))
  EPS = float(input("Enter epsilon value for action-selection: "))
  ALPHA = float(input("Enter learning rate (⍺): "))
  print("\n")
# Creating instance of MazeEscape class with (10,10) maze
  maze = MazeEscape(WIDTH,HEIGHT)

  # Eps greedy algorithm to choose action
  def eps_greedy(maze, s, w, eps):
    if np.random.random() > eps:    # if random value > eps => Exploit the knowledge
      q_list = []                   # create list of q-values
      for a in range(len(maze.direction.keys())):  # for each action calculate q_hat value
        q_value = q_hat(s, a, w)
        q_list.append(q_value)
      if sum(q_list) == 0:            # check if list has all q_hat values 0
        a = np.random.choice(q_list)  # then action will be chosen randomly to avoid choosing first action every time 
      else:
        a = np.argmax(q_list)         # else we select action with highest q_hat value
      return a
      
    else:
      a = np.random.choice(list(maze.direction.keys())) # choose action randomly => Exploring
      return a

  # Q_hat value function q_hat(s,a,w) -> dot product of feature vector and weight vector
  def q_hat(s, a, w):
    q_hat = np.dot(w, x_sa(s, a))  # q_hat(s,a,w) = w x transpose(x_sa)
    return q_hat

  # Feature vector created using state and action
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
    x[6] = s[0] + maze.direction[a][0]
    x[6] = s[1] + maze.direction[a][1]
    return x

  # Semi gradient SARSA implementation 
  def Sarsa(maze, eps, alpha, E):
    w = np.zeros((8,))      # initialize weight vector
    gamma = 0.9             # gamma value
    episode = 0             # initialize episode count
    steps_dict = {}         # dictionary to store number of steps taken for each episode
    weights_dict = {}       # dictionary to store all weights for each episode

    print("Initial State:")
    print("LEGEND: \n 1 = Agent Position || 100 = Target Position || Walls are anything outside the grid")
    print(maze.maze)
    print("\n")

    for _ in range(E):      # for each episode
      print(f"Episode:{episode}")

      steps = 0
      s = maze.reset()      # generate random agent position for each episode
      a = eps_greedy(maze, s, w, eps) # selecting action using e-greedy algorithm
      
      done = False          # done for SARSA algorithm initiated to False
      while done==False:    # Until terminal state is reached, go over the loop
        steps += 1
        s_p, r, done, _ = maze.step(a)  # calling step function for action selected
        
        # if terminal state is reached agent WINS
        if done==True:      
          w += alpha*(r - q_hat(s, a, w))*x_sa(s, a)  # update final weight value
          # print("* Target Reached *")

        # else we update weight vector until we reach terminal state
        else:
          a_p = eps_greedy(maze, s_p, w, eps)   # selecting next action 
          w += alpha*(r + gamma*q_hat(s_p, a_p, w) - q_hat(s, a, w))*x_sa(s, a) # updating weight vector
          # print(w)
          s = s_p       # updating current state with next state for next step
          a = a_p       # updating current action with next action for next step

      done = False                # again update done=True to done=False to start new episode
      steps_dict[episode] = steps # storing number of steps for each episode in dictionary
      
      # print("steps:", steps)
      # print(f"weights after episode {episode} => {w}")
      weights_dict[episode] = w.copy()                  # store final weight vector for the episode in dictionary
      episode += 1

    return weights_dict, steps_dict


  # Optimal policy and Value function
  def optimalPolicy(maze, weight):

    policy = np.zeros((WIDTH,HEIGHT), dtype=str)    # policy matrix of size as maze
    values = np.zeros((WIDTH,HEIGHT))               # value matrix of size as maze
    actions = list(maze.direction.keys())           # list of action values
    a_dict = {0:"U", 1:"R", 2:"D", 3:"L"}           # dictionary of each action with their symbol for better policy representation

    for i in range(WIDTH):
      for j in range(HEIGHT):
        q_value_list = []
        s = np.array([i,j])                         # each state in matrix

        for action in actions:                      # for each action, calculate q-hat value
          q_value = q_hat(s, action, weight)
          q_value_list.append(q_value)

        temp_max_value = np.max(q_value_list)       # find maximum q-hat value out of 4 actions
        temp_a = np.argmax(q_value_list)            

        values[i][j] = temp_max_value               # assign action value to each state in value matrix
        policy[i][j] = a_dict[temp_a]               # assign action symbol to each state in matrix
      
    return values.round(3), policy                  # value is rounded off for better visiblity


  # Calling sarsa function and optimal policy function

  sarsa = Sarsa(maze, EPS, ALPHA, EPISODES)
  weights = sarsa[0]
  steps = sarsa[1]
  optimal = optimalPolicy(maze, weights[EPISODES-1])

  print("\n")
  print(f"Optimal Policy from {EPISODES} episodes of training:")
  print("=============")
  print(f"Value Matrix: \n {optimal[0]} \n\n Policy Matrix: \n {optimal[1]} \n")
  print("<======= END OF CODE =========>")

  # Plot of number of steps per episode
  plt.figure('Episodes vs Steps')
  plt.plot(list(steps.keys()), list(steps.values()))
  plt.title("Evolution of Episodes")
  plt.xlabel("Episodes")
  plt.ylabel("No. of Steps")
  plt.show()
  
