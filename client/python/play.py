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

# #给定二维数组 返回 路径*215*len（Map）
# def find_all_paths(parsedmap: List[List[Map]]) -> List[List[List[tuple[int, int]]]]:
#     paths = []

#     player_position = None  # 保存玩家的位置

#     # 遍历 parsedmap 找到玩家的位置
#     for row_idx, row in enumerate(parsedmap):
#         for col_idx, cell in enumerate(row):
#             for obj in cell.objs:
#                 if obj.type == type:
#                     if type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
#                         if player_position == None :
#                             player_position = (row_idx, col_idx)
#                             break

#     if player_position is None:
#         raise ValueError("玩家位置未找到")

#     # 创建网格
#     matrix = [[1 for _ in range(len(parsedmap[0]))] for _ in range(len(parsedmap))]
#     for row_idx, row in enumerate(parsedmap):
#         for col_idx, cell in enumerate(row):
#             for obj in cell.objs:
#                 if obj.type == 3:  # 如果 type 为 3，表示该点不可通过
#                     matrix[row_idx][col_idx] = 0
#                 else:
#                     matrix[row_idx][col_idx] = 1

#     grid = Grid(matrix=matrix)

#     # 寻找路径
#     for row_idx, row in enumerate(parsedmap):
#         row_paths = []
#         for col_idx, _ in enumerate(row):
#             end_position = (row_idx, col_idx)
#             finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
#             path, _ = finder.find_path(grid.node(player_position[0], player_position[1]),
#                                        grid.node(end_position[0], end_position[1]), grid)
#             row_paths.append(path)

#         paths.append(row_paths)

#     return paths



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




def ParseMap(map:List[Map]) -> (List[List[Map]], List[List[List[tuple]]], List[List[List[tuple]]], tuple):
    parsedMap = [[Map() for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    paths = [[[] for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    accessableNow = [[1 for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    # accessablePotential = [[1 for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    playerPosition = None
    for grid in map:
        parsedMap[grid.x][grid.y] = grid
        for obj in grid.objs:
            if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
                playerPosition = (grid.x, grid.y)
            if obj.type == ObjType.Block or obj.type == ObjType.Bomb:
                accessableNow[grid.x][grid.y] = 0
    # an abbreviation of pathfinding grid
    pfGrid = Grid(matrix=accessableNow)
    for grid in map:
        end_position = (grid.x, grid.y)
        pfGrid.cleanup()
        finder = DijkstraFinder(diagonal_movement=DiagonalMovement.never)
        newPath, _ = finder.find_path(pfGrid.node(playerPosition[0], playerPosition[1]),
                                       pfGrid.node(end_position[0], end_position[1]), pfGrid)
        finder.
        newPath1 = [(newPath[i].x, newPath[i].y) for i in range(len(newPath))]
        paths[grid.x][grid.y] = newPath1
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
        print(paths)
        exit(0)
        requests = Play(myMap, paths, playerPosition)
        client.send(PacketReq(PacketType.ActionReq, requests))
        resp = client.recv()    
        if resp.type == PacketType.GameOver:
            gContext["gameOverFlag"] = True
            gContext["result"] = resp.data.scores[gContext["playerID"]]
            logger.info(f"game over: {gContext['gameOverFlag']}, my score: {gContext['result']}")
            break