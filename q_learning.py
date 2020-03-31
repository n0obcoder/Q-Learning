import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time, os
import sys, pdb

def q(text = ''):
    print(f'>{text}<')
    sys.exit()

style.use("ggplot")

SIZE = 8 #grid size

HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 2000  # how often to play through env visually.

start_q_table = None # None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

DISPLAY_SIZE = 800 # this should be a multiple of SIZE
IMAGE_SIZE = int(DISPLAY_SIZE/SIZE)
print('IMAGE_SIZE: ', IMAGE_SIZE)

# define image paths
img_dir = 'images'
# agent_img_path = os.path.join(img_dir, 'agent.jpeg')
# agent_img_path = os.path.join(img_dir, 'agent2.jpg')
agent_img_path = os.path.join(img_dir, 'agent3.jpg')
# reward_img_path = os.path.join(img_dir, 'reward.png')
reward_img_path = os.path.join(img_dir, 'beer.jpg')
enemy_img_path = os.path.join(img_dir, 'enemy.jpg')

# read and resize images
agent_img = cv2.imread(agent_img_path)
agent_img = cv2.resize(agent_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_AREA)
reward_img = cv2.imread(reward_img_path)
reward_img = cv2.resize(reward_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_AREA)
enemy_img = cv2.imread(enemy_img_path)
enemy_img = cv2.resize(enemy_img, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_AREA)
print('agent_img.shape, reward_img.shape, enemy_img.shape: ', agent_img.shape, reward_img.shape, enemy_img.shape)

# the dict!
resized_image_dict = {1: agent_img,
                      2: reward_img,
                      3: enemy_img}

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    print('randomly inizializing the q_table...')
    for x1 in tqdm(range(-SIZE+1, SIZE)):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                    for y2 in range(-SIZE+1, SIZE):
                        # for x3 in range(-SIZE+1, SIZE):
                        #     for y3 in range(-SIZE+1, SIZE):
                                q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
    print(f'q_table randomly initialized with {6*(2*SIZE-1)**4} value !')

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


def get_unique_spawning_location(dont_spawn_here):
    all_x, all_y = zip(*dont_spawn_here)
    blob = Blob()
    
    while (blob.x in all_x) and (blob.y in all_y):
        blob = Blob()
    
    return blob


# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []

for episode in range(HM_EPISODES):
    # make sure that agent, food and enemy have different spawning locations
    dont_spawn_here = []

    player = Blob()
    dont_spawn_here.append((player.x, player.y))
    
    food = get_unique_spawning_location(dont_spawn_here)
    dont_spawn_here.append((food.x, food.y))

    enemy = get_unique_spawning_location(dont_spawn_here)
    
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!
        player.action(action)

        # '''
        #### MAYBE ###
        # enemy.move()
        # food.move()
        ##############
        # '''

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.ones((SIZE, SIZE, 3), dtype=np.uint8)*25  # starts an rbg of our size
            # env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            # env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            # env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((DISPLAY_SIZE, DISPLAY_SIZE))  # resizing so we can see our agent in all its glory.
            
            # converting it from PIL Image to np array
            img = np.array(img)

            img [(player.x)*IMAGE_SIZE:(player.x + 1)*IMAGE_SIZE, (player.y)*IMAGE_SIZE:(player.y + 1)*IMAGE_SIZE, :] = resized_image_dict[1]
            img [(food.x)*IMAGE_SIZE:(food.x + 1)*IMAGE_SIZE, (food.y)*IMAGE_SIZE:(food.y + 1)*IMAGE_SIZE, :] = resized_image_dict[2]
            img [(enemy.x)*IMAGE_SIZE:(enemy.x + 1)*IMAGE_SIZE, (enemy.y)*IMAGE_SIZE:(enemy.y + 1)*IMAGE_SIZE, :] = resized_image_dict[3]

            cv2.imshow("shiz_just_gaat_real", img)  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(200) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(60) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
