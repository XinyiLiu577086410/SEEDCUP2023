from typing import List
from req import *
from base import *
from req import *
from resp import *
from config import config
from logger import logger
import json
import socket
import sys

# 寻路库
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.dijkstra import DijkstraFinder
from pathfinding.finder.a_star import AStarFinder


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
def Play(parsedMap: List[List[Map]], routes: List[List[List[tuple]]], playerPosition: tuple) -> List:
    '''
    ActionList = []
    ActionList += GoToSafeZone()
    ActionList += GoToItem()
    ActionList += GoToRemovableBlock()
    ActionList += PlaceBomb()
    ActionList += GotoSafeZone()
    return ActionList
    '''




# def ParseMap(map:List[Map]) -> (List[List[Map]], List[List[List[tuple]]], List[List[List[tuple]]], tuple):
def ParseMap(map:List[Map]) -> (List[List[Map]], List[List[List[tuple]]], tuple):
    parsedMap = [[Map() for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    paths = [[[] for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    accessableNow = [[1 for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    # accessablePotential = [[1 for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    playerPosition = None
    for grid in map:
        parsedMap[grid.x][grid.y] = grid
        for obj in grid.objs:
            if obj.type == ObjType.Player and obj.property.player_id != gContext["playerID"] and playerPosition is None:
                playerPosition = (grid.x, grid.y)
            if obj.type == ObjType.Block or obj.type == ObjType.Bomb:
                accessableNow[grid.x][grid.y] = 0 
    # print("accessableNow:")
    # for i in range(MapEdgeLength):
    #     for j in range(MapEdgeLength):
    #         print(accessableNow[i][j], end="  ")
    #     print("\n")
    
    # pfGrid: an abbreviation of pathfinding grid
    pfGrid = Grid(matrix=accessableNow)
    for grid in map:
        end_position = (grid.x, grid.y)
        pfGrid.cleanup()
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        newPath, _ = finder.find_path(pfGrid.node(playerPosition[1], playerPosition[0]),
                                        pfGrid.node(end_position[1], end_position[0]), pfGrid)
        '''
        def node(self, x, y) -> GridNode:
        """            ^~~^~~~~~
        get node at position
        :param x: x pos
        :param y: y pos
        :return:
        """
        return self.nodes[y][x]
                          ^~~^~~~~~
        '''
        myNewPath = [(newPath[i].y, newPath[i].x) for i in range(len(newPath))]
        paths[grid.x][grid.y] = myNewPath
    return parsedMap, paths, playerPosition


# only used in play.py
gContext = {
    "playerID": -1,
    "gameOverFlag": False,
    "result": 0,
    "gameBeginFlag": False,
}


MapEdgeLength = 15
if __name__ == "__main__":
    myMap = [[Map() for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    paths = [[[] for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    playerPosition = None
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
        myMap, paths, playerPosition = ParseMap(resp.data.map)
        
        print("map:")
        for i in range(MapEdgeLength):
            for j in range(MapEdgeLength):
                if(len(myMap[i][j].objs) == 0):
                    print("0", end="  ")
                else:
                    print("%d"%myMap[i][j].objs[0].type, end="  ")
            print("\n")
        print("paths:")
        for i in range(MapEdgeLength):
            for j in range(MapEdgeLength):
                if len(paths[i][j]) == 0:
                    print("0", end="  ")
                else:
                    print("1", end="  ")
            print("\n")
       
        requests = Play(myMap, paths, playerPosition)
        client.send(PacketReq(PacketType.ActionReq, requests))
        resp = client.recv()    
        if resp.type == PacketType.GameOver:
            gContext["gameOverFlag"] = True
            gContext["result"] = resp.data.scores[gContext["playerID"]]
            logger.info(f"game over: {gContext['gameOverFlag']}, my score: {gContext['result']}")
            break