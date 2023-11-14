import json
import socket
from base import *
from req import *
from resp import *
from config import config
from ui import UI
import subprocess
# import logging
# from threading import Thread
# from itertools import cycle, count
# from time import sleep
from logger import logger
import math
import random
from collections import namedtuple, deque

import sys
# import termios
# import tty

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# record the context of global data
gContext = {
    "playerID": -1,
    "gameOverFlag": False,
    "result": 0,
    "steps": ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"],
    "gameBeginFlag": False,
}

# from cdk
class Client(object):
    """Client obj that send/recv packet.
    """
    def __init__(self) -> None:
        self.config = config
        self.host = self.config.get("host")
        self.port = self.config.get("port")
        assert self.host and self.port, "host and port must be provided"
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._connected = False

    def connect(self):
        if self.socket.connect_ex((self.host, self.port)) == 0:
            logger.info(f"connect to {self.host}:{self.port}")
            self._connected = True
        else:
            logger.error(f"can not connect to {self.host}:{self.port}")
            exit(-1)
        return

    def send(self, req: PacketReq):
        msg = json.dumps(req, cls=JsonEncoder).encode("utf-8")
        length = len(msg)
        self.socket.sendall(length.to_bytes(8, sys.byteorder) + msg)
        # uncomment this will show req packet
        # logger.info(f"send PacketReq, content: {msg}")
        return

    def recv(self):
        length = int.from_bytes(self.socket.recv(8), sys.byteorder)
        result = b""
        while resp := self.socket.recv(length):
            result += resp
            length -= len(resp)
            if length <= 0:
                break

        # uncomment this will show resp packet
        # logger.info(f"recv PacketResp, content: {result}")
        packet = PacketResp().from_json(result)
        return packet

    def __enter__(self):
        return self
    
    def close(self):
        logger.info("closing socket")
        self.socket.close()
        logger.info("socket closed successfully")
        self._connected = False
    
    @property
    def connected(self):
        return self._connected

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        if traceback:
            print(traceback)
            return False
        return True

#from cdk
def cliGetInitReq():
    """Get init request from user input."""
    return InitReq(config.get("player_name"))


def recvAndRefresh(ui: UI, client: Client):
    """Recv packet and refresh ui."""
    global gContext
    resp = client.recv()

    if resp.type == PacketType.ActionResp:
        gContext["gameBeginFlag"] = True
        gContext["playerID"] = resp.data.player_id
        ui.player_id = gContext["playerID"]


    while resp.type != PacketType.GameOver:
        subprocess.run(["clear"])
        ui.refresh(resp.data)
        ui.display()
        resp = client.recv()

    print(f"Game Over!")

    print(f"Final scores \33[1m{resp.data.scores}\33[0m")

    if gContext["playerID"] in resp.data.winner_ids:
        print("\33[1mCongratulations! You win! \33[0m")
    else:
        print(
            "\33[1mThe goddess of victory is not on your side this time, but there is still a chance next time!\33[0m"
        )

    gContext["gameOverFlag"] = True
    print("Game Finished")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# user function
def getState(resp : PacketResp):
    for map in resp.data.map:
        if len(map.objs):
            for obj in map.objs:
                if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
                    state = [obj.property.hp, obj.property.bomb_range, obj.property.shield_time, obj.property.invincible_time]
    for map in resp.data.map:
        if len(map.objs):
            state.append(map.objs[0].type)
        else:
            state.append(0)
    return state
    # raise Exception("Invalid State")


# user function
def calcReward(resp1 : PacketResp, resp2 : PacketResp):
    score1 = 0
    score2 = 0
    for map in resp1.data.map:
        if len(map.objs):
            for obj in map.objs:
                if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
                    score1 = obj.property.score
    if resp2.type != PacketType.GameOver:
        for map in resp2.data.map:
            if len(map.objs):
                for obj in map.objs:
                    if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
                        score2 = obj.property.score
    else:
        for res in resp2.data.scores:
            if res["player_id"] == gContext["playerID"]:
                score2 = res["score"]
                gContext["result"] = res["score"]

    return score2-score1

# ref:https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions 
n_actions = 6
# Get the number of state observations
n_observations = 4 + 225

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randint(0,5)]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# user function
def termPlayAPI():
    client = Client()
    client.connect()
    initPacket = PacketReq(PacketType.InitReq, cliGetInitReq())
    client.send(initPacket)
        
    stat_resp = client.recv()

    if stat_resp.type == PacketType.ActionResp:
        gContext["gameBeginFlag"] = True
        gContext["playerID"] = stat_resp.data.player_id
    
    return client, stat_resp

direct = [[-1,0],[0,1],[1,0],[0,-1],[0,0]]
route = []

# user function
def search(ignore : list, pos : list, resp : PacketResp, type = ObjType.Player | ObjType.Item | ObjType.Block):
    if pos[0] < 0 or pos[0] >= 15 or pos[1] < 0 or pos[1] >= 15 :
        # print("Out!")
        return False
    for map in resp.data.map:
        if map.x == pos[0] and map.y == pos[1]:
            if len(map.objs):
                for obj in map.objs:
                    if obj.type == type:
                        if type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
                            continue
                        if type == ObjType.Block and obj.property.removable == False:
                            return False
                        route.append(pos)
                        return True
                    elif obj.type == ObjType.Block:
                        return False
            else:
                break
    for i in range(4):
        dpos = [pos[0]+direct[i][0], pos[1]+direct[i][1]]
        if dpos == ignore:
            return False
        # print(f"{pos[0]},{pos[1]} Search at {dpos[0]},{dpos[1]}")
        if search(pos, dpos, resp, type):
            route.append(pos)
            return True
    return False

# user function
def isConnected(resp : PacketResp, type = ObjType.Player | ObjType.Item | ObjType.Block):
    route.clear()
    for map in resp.data.map:
        if len(map.objs):
            for obj in map.objs:
                if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
                    # print(f"Start Search at {map.x},{map.y}\n")
                    return search([map.x, map.y], [map.x, map.y], resp, type)
    return False

# user function
def canMove(resp : PacketResp):
    return True

# user function
def bombPutted(resp : PacketResp):
    for map in resp.data.map:
        if len(map.objs):
            for obj in map.objs:
                if obj.type == ObjType.Player:
                    if obj.property.player_id == gContext["playerID"]:
                        if obj.property.bomb_now_num != 0:
                            return True
                        else:
                            return False
    raise Exception("Not Found!")


# user function
def inArea(resp : PacketResp):
    for map in resp.data.map:
        if len(map.objs):
            for obj in map.objs:
                if obj.type == ObjType.Bomb:
                    for i in range(obj.property.bomb_range):
                        for direction in range(5):
                            xx = map.x + direct[direction][0] * (i+1)
                            yy = map.y + direct[direction][1] * (i+1)
                            if xx >= 0 and xx < 15 and yy >=0 and yy < 15:
                                for map2 in resp.data.map:
                                    if map2.x == xx and map2.y == yy and len(map2.objs):
                                        for obj2 in map2.objs:
                                            if obj2.type == ObjType.Player and obj2.property.player_id == gContext["playerID"]:
                                                return [map.x, map.y, direction, i]
    return False

# user function
def transfer(_from : list, _to : list):
    if _from[0] < _to[0]:
        print(f"DOWN from {_from} to {_to}")
        return ActionType.MOVE_DOWN
    elif _from[0] > _to[0]:
        print(f"UP from {_from} to {_to}")
        return ActionType.MOVE_UP
    elif _from[1] < _to[1]:
        print(f"RIGHT from {_from} to {_to}")
        return ActionType.MOVE_RIGHT
    elif _from[1] > _to[1]:
        print(f"LEFT from {_from} to {_to}")
        return ActionType.MOVE_LEFT
    else:
        print(f"WARNING SILENT from {_from} to {_to}")
        return ActionType.SILENT

# user function
def gmove(action : list, resp : PacketResp):
    if len(action) == 0:
        return
    elif len(action) == 2:
        actionReq = ActionReq(gContext["playerID"], transfer(action[1], action[0]))
        actionPacket = PacketReq(PacketType.ActionReq, actionReq)
        client.send(actionPacket)
    else:
        actionReq = ActionReq(gContext["playerID"], transfer(action[len(action)-1], action[len(action)-2]))
        actionPacket = PacketReq(PacketType.ActionReq, actionReq)
        client.send(actionPacket)
        actionReq = ActionReq(gContext["playerID"], transfer(action[len(action)-2], action[len(action)-3]))
        actionPacket = PacketReq(PacketType.ActionReq, actionReq)
        client.send(actionPacket)

# user function
def checkSec(resp : PacketResp, direct : int):
    pos = []
    for map in resp.data.map:
        if len(map.objs):
            for obj in map.objs:
                if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
                    pos.extend([map.x, map.y])
    # match(direct):
    if direct == 0:
        for map in resp.data.map:
            if map.x == pos[0] and map.y == pos[1] -1: #left block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] and map.y == pos[1] +1: #right block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] -1 and map.y == pos[1]: #up block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
    elif direct == 1:
        for map in resp.data.map:
            if map.x == pos[0] -1 and map.y == pos[1]: #up block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] +1 and map.y == pos[1]: #down block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] and map.y == pos[1] +1: #right block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
    elif direct == 2:
        for map in resp.data.map:
            if map.x == pos[0] and map.y == pos[1] -1: #left block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] and map.y == pos[1] +1: #right block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] +1 and map.y == pos[1]: #down block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
    elif direct == 3:
        for map in resp.data.map:
            if map.x == pos[0] -1 and map.y == pos[1]: #up block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] +1 and map.y == pos[1]: #down block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] and map.y == pos[1] -1: #left block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
    elif direct == 4:
        for map in resp.data.map:
            if map.x == pos[0] and map.y == pos[1] -1: #left block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] and map.y == pos[1] +1: #right block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] -1 and map.y == pos[1]: #up block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
            if map.x == pos[0] +1 and map.y == pos[1]: #down block
                if not len(map.objs):
                    return [[map.x,map.y], pos]
    return []

# user function
def prePlay(client : Client, resp : PacketResp):
    freshed = False
    while not isConnected(resp, ObjType.Player):
        print("Not Connected!")
        if not canMove:
            print("Cant Move!")
            # keep silent
            actionReq = ActionReq(gContext["playerID"], ActionType.SILENT)
            actionPacket = PacketReq(PacketType.ActionReq, actionReq)
            client.send(actionPacket)
        else:
            if isConnected(resp, ObjType.Item):
                # get item
                print(f"Item! {route}")
                gmove(route, resp)
            else:
                if bombPutted(resp):
                    bomb = inArea(resp)
                    print(bomb)
                    if bomb:
                        # leave
                        move = checkSec(resp, bomb[2])
                        print(f"leave bomb {move}")
                        gmove(move, resp)
                    else:
                        # keep silent
                        actionReq = ActionReq(gContext["playerID"], ActionType.SILENT)
                        actionPacket = PacketReq(PacketType.ActionReq, actionReq)
                        client.send(actionPacket)
                else:
                    # goto bomb
                    isConnected(resp, ObjType.Block)
                    print(f"move to bomb {route}")
                    gmove(route, resp)
                    if len(route) == 4:
                        resp = client.recv()
                        freshed = True
                        actionReq = ActionReq(gContext["playerID"], ActionType.PLACED)
                        actionPacket = PacketReq(PacketType.ActionReq, actionReq)
                        client.send(actionPacket)
                        print(f"bomb putted")
                        gmove([route[2],route[1]], resp)
                    elif len(route) == 3:
                        actionReq = ActionReq(gContext["playerID"], ActionType.PLACED)
                        actionPacket = PacketReq(PacketType.ActionReq, actionReq)
                        client.send(actionPacket)
                        print(f"bomb putted")
                        resp = client.recv()
                        freshed = True
                        gmove([route[2],route[1]], resp)
                        actionReq = ActionReq(gContext["playerID"], ActionType.SILENT)
                        actionPacket = PacketReq(PacketType.ActionReq, actionReq)
                        client.send(actionPacket)
                    elif len(route) == 2:
                        actionReq = ActionReq(gContext["playerID"], ActionType.PLACED)
                        actionPacket = PacketReq(PacketType.ActionReq, actionReq)
                        client.send(actionPacket)
                        print(f"bomb putted")

        if not freshed:
            resp = client.recv()
        else:
            freshed = False
    print("Connected!\n")

if __name__ == "__main__":
    num_episodes = 1
    for i_episode in range(num_episodes):
        # Initialize the environment and get it's state
        client, resp1 = termPlayAPI()
        prePlay(client, resp1)
        resp1 = client.recv()
        state = getState(resp1)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        while not gContext["gameOverFlag"]:
            # get from model and send request
            action = select_action(state)
            actionReq = ActionReq(gContext["playerID"], action[0][0].item())
            actionPacket = PacketReq(PacketType.ActionReq, actionReq)
            client.send(actionPacket)

            resp2 = client.recv()
            if resp2.type != PacketType.GameOver:            
                observation = getState(resp2)
            else:
                gContext["gameOverFlag"] = True

            reward = calcReward(resp1, resp2)
            reward = torch.tensor([reward], device=device)
            resp1 = resp2

            terminated = gContext["gameOverFlag"]

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

        gContext["gameOverFlag"] = False
        client.close()
        print(f"Training {i_episode} result {gContext['result']}")