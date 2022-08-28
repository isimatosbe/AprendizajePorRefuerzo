import torch
import torch.nn as nn
import numpy as np
import collections

import dqn

import gym
import supersuit
from pettingzoo.atari import pong_v3

import math
import argparse
import datetime
import os
from time import time

# Inicialización de los parámetros del modelo
GAMMA = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.02
FINAL_DECAY_FRAME = 150000

MINI_BATCH = 32
MEMORY_SIZE = 10000
SYNC_FRECUENCY = 1000

LEARNING_RATE = 0.0001

# Con esta variable permitimos que el nivel de exploración del agente aumente tras cambiar de oponente en un
# entrenamiento multiagente. La idea es que tras cambiar de oponente necesitará explorar como se comporta el
# nuevo agente frente a los distintos estados del problema
MULTIAGENT_EXPLORATION_RATE = 0.9

# Clase que representa la memoria de repetición. Esta clase servirá de contenedor para las experiencias que 
# el agente vaya encontrando. Esta clase tendrá un método que permitirá extraer una muestra del tamaño necesario
# de la memoria
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)

# La clase del agente es la que más trabajo tiene, en ella definiremos las funciones que permiten tanto el
# reseteo del entorno como el poder realizar acciones en los mismos
class Agent():

    # self.buffer  : Por comódidad se almacenará la memoria de experencia dentro del agente
    # self.mode    : Determina el modo de entrenamiento que está siguiendo el agente. (1) si está entrenando
    #                contra el bot de OpenAI Gym o (2) si está realizando entrenamiento multiagente
    # self.render  : Si se visualizarán las partidas jugadas durante el entrenamiento
    # self.device  : Por defecto "cpu" aunque acepta "cuda"
    # self.env     : Por comódidad se almacenará la entorno de trabajo dentro del agente
    # self.epsilon : Por comódidad se almacenará el valor del epsilon de exploración dentro del agente
    def __init__(self, buffer, mode, sticky, render, device):
        self.exp_buffer = buffer
        self.mode = mode
        self.render = render
        self.device = device
        self.env = create_env(mode, sticky, render)
        self.epsilon = INITIAL_EPSILON
        self.reset()
    
    # La función reset permitirá reiniciar la recompensa obtenida durante la última partida así como
    # reiniciar el entorno de trabajo en el que estemos. Además guardaremos el estado (y acción) inicial
    # porque serán necesarios durante el entrenamiento
    def reset(self):
        self.reward = 0
        if self.mode == 1:
            self.prev_state = self.env.reset()
        else:
            self.env.reset()
            self.prev_state, _, _, _ = self.env.last()
            self.prev_action = self.env.action_space('first_0').sample()

    # El método step servirá para permitir al agente dar un paso en el entorno actual en el que se encuentre.
    # Esta función redigirirá al agente a la subfunción correspondiente con el modo de entrenamiento actual
    @torch.no_grad() # Para que la función no intervenga con el cálculo de los gradientes
    def step(self, net_P1, net_P2):
        if self.mode == 1:
            return self.step_bot(net_P1)
        else:
            return self.step_multiAgent(net_P1, net_P2)
    
    # La función intermedia e_greedy define como debe elegir una acción el agente correspondiente siguiendo
    # una política epsilon-voraz. Dado un estado y una red asociada a un agente esta función devuelve la 
    # acción que deberá realizar el agente correspondiente.
    @torch.no_grad() # Para que la función no intervenga con el cálculo de los gradientes
    def e_greedy(self, state, net):
        if np.random.random() < self.epsilon:
            if agent.mode == 1:
                action = self.env.action_space.sample()
            else:
                action = self.env.action_space('first_0').sample()
        else:
            state_a = np.array([state], copy=False)
            state_v = torch.tensor(state_a).to(self.device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        return action

    # step_bot es la función encargada de avanzar un paso en el caso en el que el agente este entrenando contra
    # el bot de OpenAI Gym
    @torch.no_grad() # Para que la función no intervenga con el cálculo de los gradientes
    def step_bot(self, net_P1):
        game_reward = None
        action = self.e_greedy(self.prev_state, net_P1)

        new_state, reward, done, _ = self.env.step(action)
        self.reward += reward

        exp = Experience(self.prev_state, action, reward, done, new_state)
        self.exp_buffer.append(exp)

        self.prev_state = new_state

        if done:
            game_reward = self.reward
            self.reset()

        return game_reward
    
    # step_multiAgent se encarga de realizar un paso completo en el entrenamiento multiagente, en este caso, 
    # como contamos con dos agentes, debe realizar dos pasos, uno por cada agente. Por la estructura de 
    # PettingZoo el orden siempre es 'first_0' --> 'second_0' --> Actualización del entorno y otra vez. Nos 
    # aprovechamos de conocer esta estructura para poder realizar los dos pasos necesarios para completar un
    # estado. Para esto pasamos como argumentos las dos redes asociadas a cada uno de los agentes, se decide
    # una acción haciendo uso de la función e_greedy y se realiza. Si la partida concreta ha acabado se devuelve
    # la recompensa acumulada por el agente 'first_0' y se resetea el entorno
    @torch.no_grad() # Para que la función no intervenga con el cálculo de los gradientes
    def step_multiAgent(self, net_P1, net_P2):
        game_reward = None
        
        # Turno jugador 'first_0'
        new_state_P1, reward, done_P1, _ = self.env.last()
        self.reward += reward

        exp = Experience(self.prev_state, self.prev_action, reward, done_P1, new_state_P1)
        self.exp_buffer.append(exp)

        if done_P1:
            action_P1 = None
        else:
            action_P1 = self.e_greedy(new_state_P1, net_P1)
            self.prev_state = new_state_P1
            self.prev_action = action_P1

        self.env.step(action_P1)

        if self.render:
            self.env.render()

        # Turno del jugador 'second_0'
        new_state_P2, reward_P2, done_P2, _ = self.env.last()

        if done_P2:
            game_reward = self.reward
            self.reset()
        else:
            action_P2 = self.e_greedy(new_state_P2, net_P2)
            self.env.step(action_P2)
            if self.render:
                self.env.render()

        return game_reward

# Definimos la función que se encarga de calcular el error cometido en el batch seleccionado
def calc_loss(batch, net, tgt_net, device="cpu"):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)    
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0 # False
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)

# Definimos el entorno en el que vamos a trabajar, para ello deberemos de preprocesar las observaciones
# aportadas por OpenAI Gym y PettinZoo como se ha explicado en el trabajo teórico se realizará el 
# siguiente preprocesado
# - Cambiar a blanco y negro la observación
# - Recortar para quedarnos solo con la zona de juego
# - Si estamos en el entrenamiento multiagente, obtener la imagen simétrica con los colores de los 
#   agentes cambiados. De esta forma aunque el agente aprende a actuar como el jugador 'first_0' (el de
#   la derecha) puede jugar también como 'second_0' (el de la izquieda)
# - Cambiar el tamaño de la imagen a 80x80
# - Tomar el máximo entre cada dos observaciones
# - Tomar decisiones cada 4 frames y repetir estas durante los siguientes 4 frames
# - Usar sticky actions en el entrenamiento (opcional)
# - Juntar 4 observaciones
# - Reordenar la observación para que tenga shape (4,80,80) porqué es como trabaja pytorch
# - Normalizar las observaciones para que los valores vayan de 0 a 1
# De todo esto se encarga la función create_env según los parámetros de entrenamiento que tomemos
def mirror_color(obs):
    orange = np.float32(148.433)
    green = np.float32(147.178)

    new_obs = obs
    rows, cols = obs.shape
    for j in list(range(16,20)) + list(range(140,144)):
        for i in range(cols):
            pixel = new_obs[i][j]
            if pixel == orange:
                new_obs[i][j] = green
            elif pixel == green:
                new_obs[i][j] = orange
    return np.array([row[::-1] for row in new_obs], dtype = np.float32)

def create_env(mode = 1, sticky = True, render = False):
    if mode == 1:
        if render:
            env = gym.make("PongNoFrameskip-v4", render_mode = "human")
        else:
            env = gym.make("PongNoFrameskip-v4")
    else:
        env = pong_v3.env()

    env = supersuit.observation_lambda_v0(env,
            lambda obs,_ : obs[:, :, 0] * 0.299 + obs[:, :, 1] * 0.587 + obs[:, :, 2] * 0.114)
    env = supersuit.observation_lambda_v0(env,
            lambda obs,_ : obs[34:194,:].astype(np.float32))
    if mode == 2:
        env = supersuit.observation_lambda_v0(env,
                lambda obs,_,agent : mirror_color(obs) if agent == "second_0" else obs)
    env = supersuit.resize_v1(env, 80, 80)

    env = supersuit.max_observation_v0(env, 2)
    if mode == 1:
        env = supersuit.frame_skip_v0(env, (4,4))
    else:
        env = supersuit.frame_skip_v0(env, 4)

    if sticky:
        env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

    env = supersuit.reshape_v0(env, (80,80,1))
    env = supersuit.frame_stack_v1(env, 4)
    env = supersuit.observation_lambda_v0(env,
            lambda obs,_ : np.array([obs[:,:,i] for i in range(4)]))
    env = supersuit.normalize_obs_v0(env)

    return env

# Una vez están todas las funciones definidas solo queda entrenar al modelo
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true", default=False, help="Activar el uso de cuda.")
    parser.add_argument("--mode", "-m", default=1, 
                        help="Modo de entrenamiento: (1) Entrenamiento contra bot de OpenAI Gym - Por defecto se usará este. (2) Entranamiento multiagente. (3) Entrenamiento híbrido.")
    parser.add_argument("--first", "-f", default=1, help="En el entrenamiento híbrido contra quien juega primero el agente: (1) Contra el bot de OpenAI Gym - Por defecto se usará este. (2) Contra si mismo.")
    parser.add_argument("--total_frames", default=2000000, help="Máximo número de frames. Por defecto 2M.")
    parser.add_argument("--render", action="store_true", default=False, help="Visualizar el entrenamiento, no recomendado.")
    parser.add_argument("--sticky", default=True, action="store_false", help="Deactivar el uso de sticky actions.")
    args = parser.parse_args()

    # Extraemos los parámetros del entrenamiento 
    device = torch.device("cuda" if args.cuda else "cpu")
    training = int(args.mode)
    first = int(args.first)
    TOTAL_FRAMES = int(args.total_frames)
    BACKUP_FRECUENCY = TOTAL_FRAMES // 10
    render = args.render
    sticky = args.sticky

    # Creamos una carpeta donde guardaremos la información sobre el entrenamiento (como los modelos guardados)
    # y un archivo en el que guardamos los parámetros usados y los resultados obtenidos
    training_str = "bot" if training == 1 else "multiagente" if training == 2 else "hybrid"
    save_name = "./saves/training_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")+ "/model"
    try:
        os.mkdir("./saves/")
    except:
        pass
    os.mkdir(save_name[:-5])
    f = open(save_name[:-5]+"/metadata.txt", 'w')
    f.write("Modo de entrenamiento: %s\nPrimero contra: %s\nDurante %s frames\nUsando sticky actions: %s\n"%(training_str, "bot" if training == 3 and first == 1 else "si mismo", TOTAL_FRAMES, sticky))
    f.close()

    # Inicializamos el buffer, el agente, las redes y el optimizador
    buffer = ExperienceBuffer(MEMORY_SIZE)
    agent = Agent(buffer = buffer, mode = first if training == 3 else training, sticky = sticky, render = render, device = device)

    if agent.mode == 1:
        input_shape = agent.env.observation_space.shape
        n_actions = agent.env.action_space.n
    else:
        input_shape = agent.env.observation_space('first_0').shape
        n_actions = agent.env.action_space('first_0').n

    net = dqn.DQN(input_shape, n_actions).to(device)
    tgt_net = dqn.DQN(input_shape, n_actions).to(device)
    net_2 = dqn.DQN(input_shape, n_actions).to(device)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # Inicializamos algunos valores estadísticos que necesitaremos durante el entrenamiento
    frames = 0
    # Como se ha comentado, cuando el entrenamiento sea multiagente y se cambien los rivales del agente
    # aumentaremos el valor de epsilon, esto se hará mediante el uso de frames_ma
    frames_ma = 0
    t_initial = time()
    t0 = time()
    t0_frames = 0

    rewards = []
    best_avg_reward = -math.inf

    opponents = 0
    changed_opponents = False

    f = open(save_name[:-5]+"/metadata.txt",'a')
    f.write("\nRESULTADOS\n----------\n")

    # Comenzamos el entrenamiento
    while True:
        frames += 1
        frames_ma += 1
        agent.epsilon = max(FINAL_EPSILON, INITIAL_EPSILON - frames_ma / FINAL_DECAY_FRAME)
        
        reward = agent.step(net, net_2)

        # Que el reward sea no nulo significa que se ha acabado una partida
        if reward is not None:
            rewards.append(reward)
            # Vamos calculando las medias de las últimas 100 partidas
            mean_100 = np.mean(rewards[-100:])

            speed = (frames - t0_frames) / (time() - t0)
            t0 = time()
            t0_frames = frames
            print("[%s] %d frames: %d partidas terminadas, recompensa media %.3f, eps %.2f, velocidad %.2f f/s" 
                    % (str(datetime.timedelta(seconds = t0 - t_initial))[:-7], frames, len(rewards), mean_100, agent.epsilon, speed))

            # Si estamos entrenando al agente contra el bot de OpenAI Gym guardaremos el modelo cada vez que se mejore
            # la recompensa media de los últimos 100 episodios
            if agent.mode == 1:
                if mean_100 > best_avg_reward:
                    torch.save(net.state_dict(), save_name + ".dat")
                    print("Recompensa media mejorada %.4f -> %.4f. Modelo guardado."%(best_avg_reward, mean_100))
                    best_avg_reward = mean_100

            # Si estamos ante un entrenamiento multiagente guardamos el modelo cuando este sea lo suficentemente bueno
            # contra su rival actual. Esto se ha expresado pidiendo que la recompensa media obtenida en las últimas 
            # 100 partidas sea positiva y además se hayan jugado 10 o más partidas
            else:                    
                if mean_100 > 0 and len(rewards) >= 10:
                    torch.save(net.state_dict(), save_name + ".dat")
                    net_2.load_state_dict(net.state_dict())
                    opponents += 1
                    f.write("Vector de recompensas del oponente %i: %s\n"%(opponents, rewards))
                    rewards = []
                    frames_ma *= MULTIAGENT_EXPLORATION_RATE
                    print("Acabo de vencer al oponente %i! Cambiando de oponente! Modelo guardado."%(opponents))

            # Si estamos ante un entrenamiento híbrido donde el agente jugará tanto contra el bot de OpenAI Gym como
            # contra sí mismo, necesitaremos definir el momento de cambio. En este caso se hace cuando se hayan 
            # superado la mitad del máximo de frames
            if training == 3 and frames >= TOTAL_FRAMES / 2 and not changed_opponents:
                if agent.mode == 1:
                    f.write("Mejor recompensa obtenida contra el bot de OpenAI Gym: %.4f\nVector de recompensas contra el bot de OpenAI Gym: %s\n"%(best_avg_reward, rewards))
                else:
                    f.write("Vector de recompensas del oponente %i: %s\n"%(opponents, rewards))
                agent.mode = 3 - agent.mode
                agent.env = create_env(mode = agent.mode, render = agent.render)
                agent.reset()
                net_2.load_state_dict(net.state_dict())
                best_avg_reward = -math.inf
                rewards = []
                print("Cambiando a jugar contra mi mismo.")
                changed_opponents = True

            # Si superamos el número de frames máximos guardamos el modelo y acabamos el entrenamiento
            if frames >= TOTAL_FRAMES:
                torch.save(net.state_dict(), save_name + "_final.dat")
                print("Entrenamiento terminado! Modelo guardado!\n")
                break
        
        # Cada BACKUP_FRECUENCY guardamos una copia del modelo que nos servirá para comprobar como ha mejorado
        # a lo largo del entrenamiento
        if frames % BACKUP_FRECUENCY == 0:
            torch.save(net.state_dict(), save_name + "_" + str(frames // BACKUP_FRECUENCY) + ".dat")
            print("Modelo guardado!")

        # Por último, si la memoria de experiencia está llena debemos de ir actualizando la red objetivo e 
        # ir mejorando nuestra red también
        if len(buffer) < MEMORY_SIZE:
            continue
    
        if frames % SYNC_FRECUENCY == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(MINI_BATCH)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    
    if agent.mode == 1:
        f.write("Mejor recompensa obtenida contra el bot de OpenAI Gym: %.4f\nVector de recompensas contra el bot de OpenAI Gym: %s\n"%(best_avg_reward, rewards))
    else:
        f.write("Vector de recompensas del oponente %i: %s\n"%(opponents + 1, rewards))
    f.close()