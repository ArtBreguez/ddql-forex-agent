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

print("window size", int(sys.argv[2]))

models_dir = "models/"
try:
    latest_model = max(glob.glob(models_dir + "model_ep*"), key=os.path.getctime)
except Exception:
    print("Could not find latest model")
    latest_model = None
if latest_model:
    print("Carregando modelo mais recente:", latest_model)
    episode_offset = int(latest_model.split("model_ep")[1])
    agent = Agent(window_size, model_name=latest_model)
    print("Modelo carregado com sucesso!")
else:
    episode_offset = 0
    agent = Agent(window_size)

data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32

action_limit = 5
action_count = {'hold': 0} 

for e in range(episode_offset, episode_offset + episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_offset + episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.buy_inventory = []
    agent.sell_inventory = []
    agent.bankroll = 10000

    for t in range(l):
        action = agent.act(state)
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0
        profit = 0
        action_count['hold'] = 0


        if action == 1:  # buy
            bought_price = data[t]
            print("Buy: " + formatPrice(bought_price))
            action_count['hold'] = 0
            if len(agent.sell_inventory) > 0:
                sold_price = agent.sell_inventory.pop(0)
                profit = (sold_price - bought_price) * 1000
                agent.bankroll += profit
                total_profit += profit
                print("Profit: " + formatPrice(profit))
            else:
                if agent.bankroll >= bought_price * 1000:
                    agent.bankroll -= bought_price * 1000
                    agent.buy_inventory.append(bought_price)
                else:
                    print("No enough money to buy")
                    reward += -5

        elif action == 2:  # sell
            sold_price = data[t]
            print("Sell: " + formatPrice(sold_price))
            action_count['hold'] = 0
            if len(agent.buy_inventory) > 0:
                bought_price = agent.buy_inventory.pop(0)
                profit = (sold_price - bought_price) * 1000
                agent.bankroll += profit
                total_profit += profit
                print("Profit: " + formatPrice(profit))
            else:
                if agent.bankroll >= sold_price * 1000:
                    agent.bankroll -= sold_price * 1000
                    agent.sell_inventory.append(sold_price)
                else:
                    print("No enough money to sell")
                    reward += -5

        print("Bankroll: " + formatPrice(agent.bankroll))

        elif action == 0:  # hold
            action_count['hold'] += 1
            print("Hold")

        reward += profit

        if action_count['hold'] > action_limit:
            reward += -10

        # Finalizar o episódio se o agente não puder comprar ou vender
        if len(agent.buy_inventory) == 0 and len(agent.sell_inventory) == 0 and agent.bankroll < 1500:
            done = True
        else:
            done = False

        # Finalizar o episódio se estivermos no último passo de tempo
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