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

# from typing import List, Tuple, Union
# 没有用到
# 寻路库
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.dijkstra import DijkstraFinder
from pathfinding.finder.a_star import AStarFinder


# copy from main.py
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


'''
README:
0) 在自己的分支上写代码，不要直接在main分支上写。通过pull request来合并代码
1) 可以修改你负责的函数，但是修改返回值和参数需要协商，修改函数名是不允许的
2) 你可以自己添加函数，但是需要定义在你负责的函数之内
3) 对函数的功能、参数和原型有疑问请提出
4) 可以输出信息，请在信息前加上函数名，如 
    logger.info("GoToItem(): xxxxxxx") 或 print("GoToItem(): xxxxxxx")
5) 函数以及关键要写注释，用中文。建议用Github Copilot + 人工修改。
6) 请不要修改其他人的函数，如果有需要请提出
7) 变量命名规范：小驼峰命名法， 函数命名规范：大驼峰命名法， 用英文全称，不要用拼音
8) 缩进：Space 4
9) 代码至少要进行语法测试，功能测试留到函数全部完成再进行
---------------->y
|   地图说明：
|   x 为行号
|   y 为列号
|
V
x
'''


def GoToItem(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
             playerPosition: tuple, dangerousGrids: List[tuple]) -> List[ActionReq]:
    targets = []
    for i in range(MapEdgeLength):
        for j in range(MapEdgeLength):
            for obj in parsedMap[i][j].objs:
                if obj.type == ObjType.Item:
                    targets.append((i,j))
    return safeGoTo(targets, routes, playerPosition, dangerousGrids)

#lxy
def GoToRemovableBlock(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
                          playerPosition: tuple, dangerousGrids: List[tuple]) -> List[ActionReq]:
    targets = []
    directions = [[-1,0],[0,1],[1,0],[0,-1]]
    for i in range(MapEdgeLength):
        for j in range(MapEdgeLength):
            for dir in directions:
                if i+dir[0] >= 0 and i+dir[0] < MapEdgeLength and j+dir[1] >= 0 and j+dir[1] < MapEdgeLength:
                    if len(parsedMap[i+dir[0]][j+dir[1]].objs):
                        for obj in parsedMap[i+dir[0]][j+dir[1]].objs:
                            if obj.type == ObjType.Block and obj.property.removable == True and not any(routes[i][j]) in dangerousGrids:
                                targets.append((i,j))
    if len(targets) == 0:
        return []
    return safeGoTo(targets, routes, playerPosition, dangerousGrids)


#xry
def PlaceBomb(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
                playerPosition: tuple, enemyTable: dict, dangerousGrids : List[tuple]) -> List[ActionReq]:
    '''
    参数：
        parsedMap: 解析后的map
        routes: 解析后的路径
        playerPosition: 玩家当前的位置
        enemyTable: 敌人的位置，是一个dict，key是player_id，value是一个tuple，表示坐标
    返回值：
        一个List，每个元素是一个ActionReq对象，表示一个动作请求
    功能：
        工具函数。
        如果条件成熟，返回一个放炸弹并逃走的动作请求列表
        如果玩家当前位置不可以放炸弹，返回空列表
        要做安全性检查
    '''
    def checkBombBlocked(parsedMap: List[List[Map]], gridToCheck : tuple):
        if len(parsedMap[gridToCheck[0]][gridToCheck[1]].objs):
            for obj in parsedMap[gridToCheck[0]][gridToCheck[1]].objs:
                if obj.type == ObjType.Block:
                    return True
        return False
    
    def checkBombUseful(parsedMap: List[List[Map]], gridToCheck : tuple):
        if len(parsedMap[gridToCheck[0]][gridToCheck[1]].objs):
            for obj in parsedMap[gridToCheck[0]][gridToCheck[1]].objs:
                if obj.type == ObjType.Block and obj.property.removable == True:
                    return True
        return False
    
    # 检查当前位置能否放下炸弹
    assert len(parsedMap[playerPosition[0]][playerPosition[1]].objs), "the grid where player is shouldn't be empty"
    for obj in parsedMap[playerPosition[0]][playerPosition[1]].objs:
        if obj.type == ObjType.Bomb:
            return []
    # 检查
    # a)获取炸弹范围
    bombRange = 0
    for obj in parsedMap[playerPosition[0]][playerPosition[1]].objs:
        if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
            bombRange = obj.property.bomb_range
    assert bombRange, "player's bomb range should never be zero"
    #wtf is this~~~~^~~~~~
    # 注释要用中文
    # b)检查炸弹是否能炸到砖块
    directions = [[-1,0],[0,1],[1,0],[0,-1],[0,0]]
    brickBombed = []
    for i in range(bombRange):
        for direction in directions:
            if playerPosition[0] + direction[0] * (i+1) >= 0 and playerPosition[0] + direction[0] * (i+1) < MapEdgeLength and playerPosition[1] + direction[1] * (i+1) >= 0 and playerPosition[1] + direction[1] * (i+1) < MapEdgeLength:
                gridToCheck = (playerPosition[0] + direction[0] * (i+1), playerPosition[1] + direction[1] * (i+1))
                if(checkBombBlocked(parsedMap, gridToCheck)):
                    directions.remove(direction)
                if(checkBombUseful(parsedMap, gridToCheck)):
                    brickBombed.append(gridToCheck)
    if len(brickBombed) == 0:
        return []
    
    changedMap = parsedMap.copy()
    changedMap[playerPosition[0]][playerPosition[1]].objs.append(Obj(ObjType.Bomb, Bomb(999, bombRange, gContext["playerID"])))
    escapeRoute, _ = GoToSafeZone(changedMap, routes, playerPosition)
    return [ActionReq(gContext["playerID"], ActionType.PLACED)].extend(escapeRoute)


def GoToSafeZone(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
                    playerPosition: tuple) ->(List[ActionReq], List[tuple]):
    '''
    参数：
        parsedMap: 解析后的map
        routes: 解析后的路径
        playerPosition: 玩家当前的位置
    返回值：
        一个List，每个元素是一个ActionReq对象，表示一个动作请求
        一个List，每个元素是一个tuple，表示一个危险的格子
    功能：
        工具函数。
        如果玩家在危险区域，返回一个动作请求列表，使得玩家能够逃离危险区域
        如果玩家不在危险区域，返回空列表
        同时返回一个危险的格子列表，是后续函数的参数
    '''

    def checkBombBlocked(parsedMap: List[List[Map]], gridToCheck : tuple):
        if len(parsedMap[gridToCheck[0]][gridToCheck[1]].objs):
            for obj in parsedMap[gridToCheck[0]][gridToCheck[1]].objs:
                if obj.type == ObjType.Block:
                    return True
        return False

    def CheckDangerZone(parsedMap: List[List[Map]], routes: List[List[List[tuple]]], 
                        playerPosition: tuple) -> (bool, List[tuple]):
        inDangerZone = False
        dangerousGrid = []
        directions = [[-1,0],[0,1],[1,0],[0,-1],[0,0]]
        for x in range(MapEdgeLength):
            for y in range(MapEdgeLength):
                if len(parsedMap[x][y].objs):
                    for obj in parsedMap[x][y].objs:
                        if obj.type == ObjType.Bomb:
                            # print(f"Bomb on{(x,y)}")
                            for i in range(obj.property.bomb_range):
                                for direction in directions:
                                    gridToCheck = (x + direction[0] * (i+1), y + direction[1] * (i+1))
                                    if gridToCheck[0] >= 0 and gridToCheck[0] < MapEdgeLength and gridToCheck[1] >= 0 and gridToCheck[1] < MapEdgeLength:
                                        if(checkBombBlocked(parsedMap, gridToCheck)):
                                            directions.remove(direction)
                                        dangerousGrid.append(gridToCheck)
                                        if gridToCheck == playerPosition:
                                            inDangerZone = True
        # print(dangerousGrid)
        return inDangerZone, dangerousGrid

    inDangerZone, dangerousGrid = CheckDangerZone(parsedMap, routes, playerPosition)
    if not inDangerZone:
        return [], dangerousGrid
    else:
        # now player is in danger zone
        # find the nearest safe grid (within minimum steps)
        idealRoute = [tuple() for i in range(255)]
        for x in range(MapEdgeLength):
            for y in range(MapEdgeLength):
                if not ((x,y) in dangerousGrid) and len(routes[x][y]) and len(routes[x][y]) < len(idealRoute):
                    idealRoute = routes[x][y]
        if len(idealRoute) != 255:
            return routeToActionReq(idealRoute), dangerousGrid
        else:
            # now no way to go
            print("GoToSafeZone(): No grid to go!")
            logger.warning("GoToSafeZone: No grid to go!")
            return [], dangerousGrid


def safeGoTo(targets : List[tuple], routes : List[List[List[tuple]]], playerPosition : tuple, dangerousGrid: List[List]) -> List[ActionReq]:
    '''
    参数：
        targets: 目标位置，是一个List，每个元素是一个tuple，表示坐标
        routes: 解析后的路径，是一个三维数组，每个元素是一个二维数组，每个元素是一个tuple，表示坐标
        playerPosition: 玩家当前的位置，是一个tuple，表示坐标
    返回值：
        一个List，每个元素是一个ActionReq对象，表示一个动作请求
    功能：
        工具函数。也可以自己重复写相应的过程。
        给定目标位置，返回一个动作请求列表，使得玩家能够向达最优目标位置前进
    '''
    targetPath = [tuple() for i in range(999)]
    for target in targets:
        if insideGrid(target) == False:
            print("GoTo() : Target out of range!")
            print(target)
            logger.error("GoTo(): Target out of range!")
            exit(-1)
        else:
            if not any (routes[target[0]][target[1]]) in dangerousGrid:
                if len(routes[target[0]][target[1]]) and len(routes[target[0]][target[1]]) < len(targetPath):
                    targetPath = routes[target[0]][target[1]]
    if len(targetPath) == 999:
        print("GoTo() : No route to go!")
        logger.warning("GoTo(): No route to go!")
        return []
    else:
        print("Goto() : Heading to " + str(targetPath[-1]))
        return routeToActionReq(targetPath)
        

def routeToActionReq(route: List[tuple]) -> List[ActionReq]:
    '''
    参数：
        route: 一个List，每个元素是一个tuple，表示坐标
    返回值：
        一个List，每个元素是一个ActionReq对象，表示一个动作请求
    功能：
        工具函数。
        将一个坐标的路径转换成一个动作请求列表
    '''
    step = [ActionReq(0, 0) for i in range(len(route))]
    for i in range(1, len(route)):
        step[i] = (route[i][0] - route[i-1][0], route[i][1] - route[i-1][1])
        if step[i][0] == 0 and step[i][1] == 1:
            step[i] = ActionType.MOVE_RIGHT
        elif step[i][0] == 0 and step[i][1] == -1:
            step[i] = ActionType.MOVE_LEFT
        elif step[i][0] == 1 and step[i][1] == 0:
            step[i] = ActionType.MOVE_DOWN
        elif step[i][0] == -1 and step[i][1] == 0:
            step[i] = ActionType.MOVE_UP
        else:
            logger.error("Wrong step!")
    return [ActionReq(gContext["playerID"], x) for x in step]


def  insideGrid(grid : tuple) -> bool:
    return grid[0] >= 0 and grid[0] < MapEdgeLength and grid[1] >= 0 and grid[1] < MapEdgeLength


def Play(parsedMap: List[List[Map]], routes: List[List[List[tuple]]], playerPosition: tuple, enemyTable : dict) -> List[ActionReq]:
    '''
    参数：
        parsedMap: 解析后的map，是一个二维数组，每个元素是一个Map对象
        routes: 解析后的路径，是一个三维数组，每个元素是一个二维数组，每个元素是一个tuple，表示坐标
        playerPosition: 玩家当前的位置，是一个tuple，表示坐标
        enemyTable: 敌人的位置，是一个dict，key是player_id，value是一个tuple，表示坐标
    返回值：
        一个List，每个元素是一个ActionReq对象，表示一个动作请求
    功能：
        主函数。
        根据当前的游戏状态，返回一个动作请求列表，使得玩家能够在当前回合中达到最优状态
    '''
    actionReqList = [] # 要返回的动作请求列表
    tmpReqList, dangerousGrids =  GoToSafeZone(parsedMap, routes, playerPosition) # 先去安全区域，如果在安全区域则返回空列表，dangerousGrids表示危险的格子，是后续函数的参数
    actionReqList += tmpReqList 
    tmpReqList = GoToItem(parsedMap, routes, playerPosition, dangerousGrids) # 去道具
    actionReqList += tmpReqList
    tmpReqList = GoToRemovableBlock(parsedMap, routes, playerPosition, dangerousGrids) # 去可炸方块
    actionReqList += tmpReqList
    tmpReqList = PlaceBomb(parsedMap, routes, playerPosition, enemyTable, dangerousGrids) # 放炸弹, 并逃走
    actionReqList += tmpReqList
    print(len(actionReqList))
    return actionReqList


def ParseMap(map:List[Map]) -> (List[List[Map]], List[List[List[tuple]]], tuple, dict):
    '''
    参数：
        map: 服务器传来的map
    返回值：
        第一个值：解析后的map，是一个二维数组，每个元素是一个Map对象，代表一个格子
        第二个值：解析后的路径，是一个三维数组，每个元素是一个二维数组，代表从当前我的位置到该格子的最短路径
        第三个值：我的当前的位置，是一个tuple，表示坐标
        第四个值：敌人的位置，是一个dict，key是player_id，value是一个tuple，表示坐标
    功能：
        工具函数。
        将服务器传来的map解析成容易使用的数据结构，方便后续的操作。
        方便随机访问和搜索。
    '''
    parsedMap = [[Map() for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    paths = [[[] for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    accessableNow = [[1 for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    # accessablePotential = [[1 for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    myPosition = None
    enemyTable = {}
    myPosition = None
    enemyTable = {}
    for grid in map:
        parsedMap[grid.x][grid.y] = grid
        for obj in grid.objs:
            if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"] and myPosition is None:
                myPosition = (grid.x, grid.y)
            if obj.type == ObjType.Player and obj.property.player_id != gContext["playerID"]:
                enemyTable[obj.property.player_id] = (grid.x, grid.y)
            if obj.type == ObjType.Block or obj.type == ObjType.Bomb:
                accessableNow[grid.x][grid.y] = 0 
    pfGrid = Grid(matrix=accessableNow)
    for grid in map:
        endPosition = (grid.x, grid.y)
        pfGrid.cleanup()
        finder = AStarFinder(diagonal_movement=DiagonalMovement.never)
        newPath, _ = finder.find_path(pfGrid.node(myPosition[1], myPosition[0]),
                                      pfGrid.node(endPosition[1], endPosition[0]), pfGrid)

                                      #reversed order here
        myNewPath = [(newPath[i].y, newPath[i].x) for i in range(len(newPath))]
        paths[grid.x][grid.y] = myNewPath
    return parsedMap, paths, myPosition, enemyTable


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
        myMap, paths, playerPosition, enemyTable = ParseMap(resp.data.map)
        requests = Play(myMap, paths, playerPosition, enemyTable)
        client.send(PacketReq(PacketType.ActionReq, requests))
        resp = client.recv()    
        if resp.type == PacketType.GameOver:
            gContext["gameOverFlag"] = True
            gContext["result"] = resp.data.scores[gContext["playerID"]]
            logger.info(f"game over: {gContext['gameOverFlag']}, my score: {gContext['result']}")
            break