# Q-Learning
Agent learns to play a simple game using Q-Learning in Numpy

## Requirements
* PIL >= 6.2.0
* opencv-python >= 4.1.1 
* numpy >= 1.7.3       
* matplotlib >= 3.1.1 
* tqdm 

## Training the RL Agent
```
python q_learning.py
```

Note that if you want to train the agent from scratch (initiaalize the q-table randomly), then set **q_table** to **None** in q_learning.py, else set q_table to the path of the already saved q_table. 


## Results
Following is the plot for the moving average of the rewards. It's upward trend shows that the agent becomes smarter with more and more episodes of training.
<br>
<img src='/results/reward_vs_episode.jpg' width='350' alt='reward_vs_episode.jpg' hspace='270'>

And here are some GIFs that show how the agent gets smarter with every episode of training.
<br>
Here is the thirsty agent looking for the bottle of beer with randomly initialized q-table. It means that the agent has no clue about the environment yet.
<br>
<img src='/results/dumb_agent_gif.gif' width='350' alt='dumb_agent_gif.gif' hspace='270'>

After some training, the agent does a relatively better job of making sequential decisions. He is not very fast yet but he ends up finding the beer eventually.
<br>
<img src='/results/moderately_smart_agent_gif.gif' width='350' alt='moderately_smart_agent_gif.gif' hspace='270'>

Finally after thousands of episode of training, the agent gets really good at making sequential decisions and finds the beer in no time ! : D 
<br>
<img src='/results/smart_agent_gif.gif' width='350' alt='smart_agent_gif.gif' hspace='270'>
