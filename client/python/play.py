from typing import List
from req import *

def GotToSafeZone() -> List[ActionReq]:
    return []
def GoToItem() -> List[ActionReq]:
    return None
def GoToRemovableBlock() -> List[ActionReq]:
    return None
def PlaceBomb() -> List[ActionReq]:
    return None
def GoToSafeZone() -> List[ActionReq]:
    return None
def Play(parsedMap: List[List]) -> List:
    '''
    ActionList = []
    ActionList += GoToSafeZone()
    ActionList += GoToItem()
    ActionList += GoToRemovableBlock()
    ActionList += PlaceBomb()
    ActionList += GotoSafeZone()
    return ActionList
    '''

from base import *
from req import *
from resp import *
from config import config
from logger import logger
import json
import socket
import sys

#寻路库
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
#

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

#给定二维数组 返回 路径*215*len（Map）
def find_all_paths(map_grid: List[List[Map]]) -> List[List[List[tuple[int, int]]]]:
    paths = []

    

    for start_x in range(len(map_grid)):
        row_paths = []
        for start_y in range(len(map_grid[0])):
            start = map_grid[start_x][start_y]
            
            # 创建网格
            matrix = [[1 for pnt in range(len(map_grid[0]))] for pnt in range(len(map_grid))]
            for x in range(len(map_grid)):
                for y in range(len(map_grid[0])):
                    if any(obj.type == 3 for obj in map_grid[x][y].objs):
                        matrix[x][y] = 0
                    else:
                        matrix[x][y] = 1

            grid = Grid(matrix=matrix)

            # 寻找路径
            path_data = []
            for end_x in range(len(map_grid)):
                for end_y in range(len(map_grid[0])):
                    end = map_grid[end_x][end_y]
                    finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
                    path, pnt = finder.find_path(grid.node(start.x, start.y), grid.node(end.x, end.y), grid)
                    path_data.append(path)

            row_paths.append(path_data)

        paths.append(row_paths)

    return paths

def ParseMap(map:List[Map]) -> (List[List[Map]], List[List[List[tuple[int,int]]]]):
    parsedMap = [[Map() for i in range(Nmap)] for j in range(Nmap)]
    for grid in map:
        parsedMap[grid.x][grid.y] = grid
    
    paths = find_all_paths(parsedMap)
    return parsedMap, paths


# only used in play.py
gContext = {
    "playerID": -1,
    "gameOverFlag": False,
    "result": 0,
    "gameBeginFlag": False,
}


Nmap = 15
MyMap = [[Map() for i in range(Nmap)] for j in range(Nmap)]
if __name__ == "__main__":
    # init game
    client = Client()
    client.connect()
    initPacket = PacketReq(PacketType.InitReq, cliGetInitReq())
    client.send(initPacket)
    resp = client.recv() # blocking communication
    if resp.type == PacketType.ActionResp:
        gContext["gameBeginFlag"] = True
        gContext["playerID"] = resp.data.player_id
    else :
        logger.error("init failed")
        exit(-1)
    while(not gContext["gameOverFlag"]):
        MyMap = ParseMap(resp.data.map)
        requests = Play(MyMap)
        for req in requests:
            client.send(PacketReq(PacketType.ActionReq, req))
        resp = client.recv()    
        if resp.type == PacketType.GameOver:
            gContext["gameOverFlag"] = True
            gContext["result"] = resp.data.scores[gContext["playerID"]]
            logger.info(f"game over: {gContext['gameOverFlag']}, my score: {gContext['result']}")
            break