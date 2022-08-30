#!/usr/bin/env python3
import torch
import numpy as np
import collections

import main
import dqn

import gym
import supersuit
from pettingzoo.atari import pong_v3

import math
import argparse
import datetime
import os
import time

FPS = 25

# La función e_greedy la definimos igual que en main.py con la particularidad de que
# ahora uno de los modos puede ser totalmente aleatorio
def e_greedy(state, net, epsilon, mode ="", device = "cpu"):
    if np.random.random() < epsilon or net is None:
        if mode == "bot":
            action = env.action_space.sample()
        else:
            action = env.action_space('first_0').sample()
    else:
        state_a = np.array([state], copy=False)
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())
    return action

# Las funciones que antes determinaban un paso en una partida ahora podemos redefinirlas
# para que definan una partida completa
def play_bot(net, epsilon, device):
    state = env.reset()
    game_reward = 0

    while True:
        action = e_greedy(state, net, epsilon, "bot", device)
        new_state, reward, done, _ = env.step(action)
        game_reward += reward

        if done:
            break
        
        state = new_state
    
    return game_reward

def play_multiAgent(net_P1, net_P2, epsilon, render, device):
    env.reset()
    game_reward = 0

    while True:
        start_ts = time.time()
        # Turno 'first_0'
        state_P1, reward, done_P1, _ = env.last()
        game_reward += reward

        if done_P1:
            action_P1 = None
        else:
            action_P1 = e_greedy(state_P1, net_P1, epsilon, device = device)
        env.step(action_P1)
        if render:
            env.render()
            delta = 1/FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)

        # Turno 'second_0'
        state_P2, _, done_P2, _ = env.last()

        if done_P2:
            return game_reward
        else:
            action_P2 = e_greedy(state_P2, net_P2, epsilon, device = device)
            env.step(action_P2)
            if render:
                env.render()
                delta = 1/FPS - (time.time() - start_ts)
                if delta > 0:
                    time.sleep(delta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", required=True, help="Ruta al modelo del agente 1.")
    parser.add_argument("-m2", "--model_2", default=None, 
                        help="""Modelo contricante: Si no se especifica nada se tomará un agente aleatorio, 
                                                    'bot' será jugar contra el bot de OpenAI Gym. 
                                                    Si se quiere jugar contra otro modelo introducir la ruta al modelo del agente (este actuará como el jugador 'second_0').""")
    parser.add_argument("--eps", default=0.05, help="Valor de exploración para la política a seguir por los agentes. Por defecto 0.05.")
    parser.add_argument("-g", "--games", default=100, help="Número de partidas a jugar, por defecto 100.")
    parser.add_argument("--render", default=False, action='store_true', help="Activar visualización de las partidas.")
    parser.add_argument("--sticky", default=True, action='store_false', help="Usar sticky actions en la creación del entorno. Por defecto sí.")
    parser.add_argument("--cuda", default=False, action='store_true', help="Permitir el uso de CUDA.")
    args = parser.parse_args()

    epsilon = np.float32(args.eps)
    games = int(args.games)
    render = args.render
    mode = 0 if args.model_2 == 'bot' else 1 if args.model_2 is None else 2
    sticky = args.sticky
    device = torch.device("cuda" if args.cuda else "cpu")

    # Igual que en main.py creamos un directorio y un archivo para guardar la información importante
    # sobre las partidas
    try:
        os.mkdir("./evals/")
    except:
        pass
    file_name = "./evals/eval-" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M") + ".txt"
    
    f = open(file_name, 'w')
    f.write("Modelo 1 (jugador verde): %s\nModelo 2 (jugador naranja): %s\nEpsilon: %f\nNúmero de partidas: %i\nSticky actions: %s\n\nResultados\n----------\n"
            %(args.model, args.model_2 if args.model_2 is not None else "aleatorio", epsilon, games, sticky))
    f.close()

    # Creamos el entorno con la función que definimos en main.py y cargamos las redes
    env = main.create_env(int(mode != 0) + 1, sticky, render)

    if args.model_2 == 'bot':
        input_shape = env.observation_space.shape
        n_actions = env.action_space.n
    else:
        input_shape = env.observation_space('first_0').shape
        n_actions = env.action_space('first_0').n

    net = dqn.DQN(input_shape, n_actions).to(device)
    net.load_state_dict(torch.load(args.model, map_location=lambda storage, loc: storage))
    if args.model_2 != 'bot' and args.model_2 is not None:
        net_2 = dqn.DQN(input_shape, n_actions).to(device)
        net_2.load_state_dict(torch.load(args.model_2, map_location=lambda storage, loc: storage))

    rewards = [] 
    t_initial = time.time()
    reverse = 0

    f = open(file_name, 'a')
    # Con todo definido podemos poner al programa a jugar las partidas necesarias. Tiene la peculiaridad
    # de que cuando la partida la juegan dos agentes (una partida multiagente) hay un 50% de probabilidad
    # de que sean cualquiera de los jugadores, no siempre serán 'first_0' o 'second_0'. Esto es interesante
    # porque en muchas ejecuciones, sobre todo contra el agente aleatorio, como el jugador 'second_0' es el
    # primero en recibir el saque, si este no llegaba a darle entra en un bucle en el que pierde todas las
    # partidas igual, a partir del saque. Cambiandolos de orden elimina la posibilidad de que el modelo que
    # juegue como 'first_0' tenga ventaja.
    for i in range(games):
        if args.model_2 == 'bot':
            reward = play_bot(net, epsilon, device)
        elif args.model_2 is None:
            if np.random.random() < 0.5:
                print("Jugando como 'first_0', jugador de la derecha (verde).")
                reward = play_multiAgent(net, None, epsilon, render, device)
            else:
                print("Jugando como 'second_0', jugador de la izquierda (naranja).")
                f.write("<-> ")
                reward = -play_multiAgent(None, net, epsilon, render, device)
                reverse += 1
        else:
            if np.random.random() < 0.5:
                print("Jugando como 'first_0', jugador de la derecha (verde).")
                reward = play_multiAgent(net, net_2, epsilon, render, device)
            else:
                print("Jugando como 'second_0,' jugador de la izquierda (naranja).")
                f.write("<-> ")
                reward = -play_multiAgent(net_2, net, epsilon, render, device)
                reverse += 1
        rewards.append(reward)

        t0 = time.time()
        print("[%s] Recompensa obtenida en la partida %i: %i"%(str(datetime.timedelta(seconds = t0 - t_initial))[:-7],i+1, reward))
        f.write("Recompensa obtenida en la partida %i: %i\n"%(i+1,reward))

    print("Total reward: %.2f" %(np.mean(rewards)))
    f.write("\nRecompensa media obtenida: %.4f"%(np.mean(rewards)))
    if args.model_2 != "bot":
        print("Habiendo jugado %i partidas como Jugador 1 y %i partidas como Jugador 2."%(games - reverse, reverse))
        f.write("\nPartidas como jugador 2: %i"%reverse)
    f.close()

    if render:
        env.close()

