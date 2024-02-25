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

# Define o limite para o número de vezes que o modelo pode realizar a ação "hold" antes de ser punido
hold_limit = 5
hold_count = 0  # Contador para acompanhar quantas vezes o modelo realizou a ação "hold" consecutivamente

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []

    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:  # buy
            if len(agent.sell_inventory) > 0:
                sold_price = agent.sell_inventory.pop(0)
                reward = max((data[t] - sold_price) * 1000, 0)
                total_profit += (data[t] - sold_price) * 1000
                print("Buy: " + formatPrice(data[t]) + " | Profit: " + formatPrice((data[t] - sold_price) * 1000))
            else:
                agent.buy_inventory.append(data[t])
                print("Buy: " + formatPrice(data[t]))
                hold_count = 0  # Resetar o contador quando uma nova ação é realizada

        elif action == 2:  # sell
            if len(agent.buy_inventory) > 0:
                bought_price = agent.buy_inventory.pop(0)
                reward = max((data[t] - bought_price) * 1000, 0)
                total_profit += (data[t] - bought_price) * 1000
                print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice((data[t] - bought_price) * 1000))
            else:
                agent.sell_inventory.append(data[t])
                print("Sell: " + formatPrice(data[t]))
                hold_count = 0  # Resetar o contador quando uma nova ação é realizada

        elif action == 0:  # hold
            hold_count += 1  # Incrementar o contador quando a ação "hold" é realizada
            if hold_count >= hold_limit:
                reward -= 10  # Aplicar uma punição ao modelo por realizar a ação "hold" muitas vezes seguidas
                print("Hold: Punished!")
            print("Hold")

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
