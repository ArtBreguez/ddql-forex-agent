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

# Define o limite para o número de vezes que o modelo pode realizar a mesma ação consecutivamente
action_limit = 5
action_count = {'buy': 0, 'sell': 0, 'hold': 0}  # Dicionário para acompanhar quantas vezes o modelo realizou cada ação consecutivamente

max_loss_per_episode = -200

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []
    episode_step = 0

    for t in range(l):
        action = agent.act(state)
        print("T:", t)
        print("L:", l)
        print("episode step:", episode_step)
        next_state = getState(data, t + 1, window_size + 1)
            
        reward = 0

        if action == 1:  # buy
            action_count['sell'] = 0
            action_count['hold'] = 0
            if len(agent.sell_inventory) == 0:
                action_count['buy'] += 1
            if action_count['buy'] > action_limit:
                reward -= 10  # Punir se exceder o limite de compras consecutivas
                print("Buy: Punished for exceeding buy limit!")
            bought_price = data[t]
            agent.buy_inventory.append(bought_price)
            print("Buy: " + formatPrice(bought_price))
            if len(agent.sell_inventory) > 0:
                sold_price = agent.sell_inventory.pop(0)
                profit = (sold_price - bought_price) * 1000
                total_profit += profit
                print("Profit: " + formatPrice(profit))

        elif action == 2:  # sell
            action_count['buy'] = 0
            action_count['hold'] = 0
            if len(agent.buy_inventory) == 0:
                action_count['sell'] += 1
            if action_count['sell'] > action_limit:
                reward -= 10  # Punir se exceder o limite de vendas consecutivas
                print("Sell: Punished for exceeding sell limit!")
            sold_price = data[t]
            agent.sell_inventory.append(sold_price)
            print("Sell: " + formatPrice(sold_price))
            if len(agent.buy_inventory) > 0:
                bought_price = agent.buy_inventory.pop(0)
                profit = (sold_price - bought_price) * 1000
                total_profit += profit
                print("Profit: " + formatPrice(profit))

        elif action == 0:  # hold
            action_count['buy'] = 0
            action_count['sell'] = 0
            action_count['hold'] += 1
            if action_count['hold'] > action_limit:
                reward -= 10  # Punir se exceder o limite de holds consecutivos
                print("Hold: Punished for exceeding hold limit!")
            print("Hold")

        done = True if t == l - 1 or total_profit < max_loss_per_episode else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state
        episode_step += 1

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")
            break

        if len(agent.memory) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))
