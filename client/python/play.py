import json
import socket
from base import *
from req import *
from resp import *
from config import config
from ui import UI
import subprocess
import logging
from time import sleep
from logger import logger
import math
import random
from collections import namedtuple, deque

import sys

# record the context of global data
gContext = {
    "playerID": -1,
    "gameOverFlag": False,
    "result": 0,
    "gameBeginFlag": False,
}


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


def cliGetInitReq():
    """Get init request from user input."""
    return InitReq(config.get("player_name"))

direct = [[-1,0],[0,1],[1,0],[0,-1],[0,0]]
route = []

# user func
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

# user func
def isConnected(resp : PacketResp, type = ObjType.Player | ObjType.Item | ObjType.Block):
    route.clear()
    for map in resp.data.map:
        if len(map.objs):
            for obj in map.objs:
                if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
                    # print(f"Start Search at {map.x},{map.y}\n")
                    return search([map.x, map.y], [map.x, map.y], resp, type)
    return False

# user func
def canMove(resp : PacketResp):
    return True

# user func
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


# user func
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

# user func
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

# user func
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

# user func
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

if __name__ == "__main__":
    client = Client()
    client.connect()
    initPacket = PacketReq(PacketType.InitReq, cliGetInitReq())
    client.send(initPacket)
        
    resp = client.recv()

    if stat_resp.type == PacketType.ActionResp:
        gContext["gameBeginFlag"] = True
        gContext["playerID"] = stat_resp.data.player_id

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