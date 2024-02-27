import numpy as np
import sys
from agent.agent import Agent
from functions import *

if len(sys.argv) != 4:
    print("Usage: python train.py [stock] [window] [episodes]")
    exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

print("window size", int(sys.argv[2]))

agent = Agent(window_size)
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

action_limit = 7
action_count = {'buy': 0, 'sell': 0, 'hold': 0} 

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.buy_inventory = []
    agent.sell_inventory = []

    for t in range(l):
        action = agent.act(state)
		next_state = getState(data, t + 1, window_size + 1)
            
        reward = 0

        if action == 1:  # buy
            bought_price = data[t]
            print("Buy: " + formatPrice(bought_price))
            action_count['sell'] = 0
            action_count['hold'] = 0
            if len(agent.sell_inventory) > 0:
                sold_price = agent.sell_inventory.pop(0)
                profit = (sold_price - bought_price) * 1000
                total_profit += profit
                print("Profit: " + formatPrice(profit))
            else:
                agent.buy_inventory.append(bought_price)
            action_count['buy'] += 1

        elif action == 2:  # sell
            sold_price = data[t]
            print("Sell: " + formatPrice(sold_price))
            action_count['buy'] = 0
            action_count['hold'] = 0
            if len(agent.buy_inventory) > 0:
                bought_price = agent.buy_inventory.pop(0)
                profit = (sold_price - bought_price) * 1000
                total_profit += profit
                print("Profit: " + formatPrice(profit))
            else:
                agent.sell_inventory.append(sold_price)
            action_count['sell'] += 1

        elif action == 0:  # hold
            action_count['hold'] += 1
            print("Hold")

        reward = profit

        if action_count['buy'] > action_limit || action_count['hold'] > action_limit || action_count['sell'] > action_limit
            reward =+ -10

        done = True if t == l - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))
