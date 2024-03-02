import numpy as np
import sys
import os
import glob
from agent.agent import Agent
from functions import *

if len(sys.argv) != 4:
    print("Usage: python train.py [stock] [window] [episodes]")
    exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

models_dir = "models/"
try:
    latest_model = max(glob.glob(models_dir + "model_ep*"), key=os.path.getctime)
except Exception:
    print("Could not find latest model")
    latest_model = None
if latest_model:
    print("Latest model:", latest_model)
    episode_offset = int(latest_model.split("model_ep")[1])
    agent = Agent(window_size, model_name=latest_model)
    print("Latest Model Loaded!")
else:
    episode_offset = 0
    agent = Agent(window_size)

data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

action_limit = 5
action_count = {'hold': 0}
transaction_fee = 0.1 # 0.1$ taxa de transação

for e in range(episode_offset, episode_offset + episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_offset + episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.buy_inventory = []
    agent.sell_inventory = []
    agent.bankroll = 10000
    action_count['hold'] = 0

    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        profit = 0

        if action == 1:  # buy
            bought_price = data[t]
            print("Buy: " + formatPrice(bought_price[3]))
            action_count['hold'] = 0
            if len(agent.sell_inventory) > 0:
                sold_price = agent.sell_inventory.pop(0)
                profit = ((sold_price - bought_price[3]) * 1000) - transaction_fee
                agent.bankroll += profit + (bought_price[3] * 1000) 
                total_profit += profit
                print("Profit: " + formatPrice(profit))
            else:
                if agent.bankroll >= bought_price[3] * 1000:
                    agent.bankroll -= bought_price[3] * 1000
                    agent.buy_inventory.append(bought_price[3])
                else:
                    print("No enough money to buy")
                    reward += -5

        elif action == 2:  # sell
            sold_price = data[t]
            print("Sell: " + formatPrice(sold_price[3]))
            action_count['hold'] = 0
            if len(agent.buy_inventory) > 0:
                bought_price = agent.buy_inventory.pop(0)
                profit = ((sold_price[3] - bought_price) * 1000) - transaction_fee
                agent.bankroll += profit + (sold_price[3] * 1000)
                total_profit += profit
                print("Profit: " + formatPrice(profit))
            else:
                if agent.bankroll >= sold_price[3] * 1000:
                    agent.bankroll -= sold_price[3] * 1000
                    agent.sell_inventory.append(sold_price[3])
                else:
                    print("No enough money to sell")
                    reward += -5

        elif action == 0:  # hold
            action_count['hold'] += 1
            print("Hold count: " + str(action_count['hold']))

        print("Bankroll: " + formatPrice(agent.bankroll))

        reward += profit

        if action_count['hold'] > action_limit:
            reward += -(action_count['hold'])

        if len(agent.buy_inventory) == 0 and len(agent.sell_inventory) == 0 and agent.bankroll < 1500:
            done = True
        else:
            done = False

        if t == l - 1:
            done = True

        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("Bankroll: " + formatPrice(agent.bankroll))
            print("--------------------------------")
            break

    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))