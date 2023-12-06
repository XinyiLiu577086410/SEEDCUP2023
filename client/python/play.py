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
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
# from pathfinding.finder.dijkstra import DijkstraFinder
from pathfinding.finder.a_star import AStarFinder

attackDistance = 3 # 攻击距离, 应避免太大
ChaseDistance = 7
maxSpeed = 2# 移动速度
hasGloves = False# 能否推动炸弹
gContext = {
    "playerID": -1,
    "gameOverFlag": False,
    "result": 0,
    "gameBeginFlag": False,
}
MapEdgeLength = 19

# copy from main.py (a part of SDK demo)
class Client(object):
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
    return InitReq(config.get("player_name"))


def GoToItem(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
             playerPosition: tuple, dangerousGrids: List[tuple]) -> List[ActionReq]:
    targets = []
    for i in range(MapEdgeLength):
        for j in range(MapEdgeLength):
            for obj in parsedMap[i][j].objs:
                if obj.type == ObjType.Item:
                    targets.append((i,j))
    print("GoToItem(): calling safeGoTo()")
    return safeGoTo(targets, routes, playerPosition, dangerousGrids)


def NextToRemovableBlock(parsedMap: List[List[Map]], playerPosition: tuple) -> bool:
    directions = [[-1,0],[0,1],[1,0],[0,-1]]
    # print("NextToRemovableBlock(): ", end="")
    # print(playerPosition)
    for dir in directions:
        GridToCheck = (playerPosition[0]+dir[0], playerPosition[1]+dir[1])
        if insideGrids(GridToCheck):
            for obj in parsedMap[GridToCheck[0]][GridToCheck[1]].objs:
                if obj.type == ObjType.Block and obj.property.removable == True:
                    # print(obj)
                    # print(gridToCheck)
                    # print("NextToRemovableBlock(): True")
                    return True
    return False


def GoToRemovableBlockAndPlaceBomb(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
                          playerPosition: tuple, dangerousGrids: List[tuple]) -> List[ActionReq]:
    targets = []
    directions = [[-1,0],[0,1],[1,0],[0,-1]]
    for i in range(MapEdgeLength):
        for j in range(MapEdgeLength):
            # 附近有可炸的砖块
            if NextToRemovableBlock(parsedMap, playerPosition):
                return PlaceBomb(parsedMap, routes, playerPosition, dangerousGrids)
            if playerPosition == (i,j):
                continue
            for dir in directions:
                GridToCheck = (i+dir[0], j+dir[1])
                if insideGrids(GridToCheck):
                    for obj in parsedMap[i+dir[0]][j+dir[1]].objs:
                        if obj.type == ObjType.Block and obj.property.removable == True:
                            targets.append((i,j))
    print("GoToRemoveableBlock(): calling safeGoTo()")
    return safeGoTo(targets, routes, playerPosition, dangerousGrids)


def PlaceBomb(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
                playerPosition: tuple, dangerousGrids : List[tuple]) -> List[ActionReq]:
    
    # 检查当前位置能否放下炸弹
    for obj in parsedMap[playerPosition[0]][playerPosition[1]].objs:
        if obj.type == ObjType.Bomb:
            return []
    # 检查
    # a)获取炸弹范围
    bombRange = 0
    for obj in parsedMap[playerPosition[0]][playerPosition[1]].objs:
        if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
            bombRange = obj.property.bomb_range    
    # b)检查放完炸弹后的安全性
    changedMap = parsedMap.copy()
    changedMap[playerPosition[0]][playerPosition[1]].objs.append(Obj(ObjType.Bomb, Bomb(999, bombRange, gContext["playerID"])))
    for i in range(MapEdgeLength):
        for j in range(MapEdgeLength):
           for step in routes[i][j]:
               if step in dangerousGrids:
                   routes[i][j] = [] 
                   break                  

    _, desperate, _ = AnalyseDanger(changedMap, playerPosition, routes)
    if desperate:
        return []
    else:
        print("PlaceBomb(): Bomb placed at " + str(playerPosition))
        return [ActionReq(gContext["playerID"], ActionType.PLACED)]
    

# 仅当位于危险区域时调用
# 会冒险通过危险区域逃生
def EscapeToSafeZone(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
                    playerPosition: tuple, dangerousGrids: List[List[tuple]]) -> List[ActionReq]: 
    EscapeRoute = ChooseEscapeRoute(routes, playerPosition, dangerousGrids)
    if len(EscapeRoute) == 0:
        print("GoToSafeZone(): I am so Desperate!")
    else:
        print("GoToSafeZone(): Escaping to " + str(EscapeRoute[-1]))
    return routeToActionReq(EscapeRoute) + [ActionReq(gContext["playerID"], ActionType.SILENT)] # 避免后续的冒险行为


def ChooseEscapeRoute(routes : List[List[List[tuple]]], 
                          playerPosition : tuple, dangerousGrids : List[tuple]) -> List[tuple]:
    EscapeRoute = [tuple() for i in range(255)]
    for x in range(MapEdgeLength):
        for y in range(MapEdgeLength):
            # 注意 route 为空的情况表示不可达
            if not ((x,y) in dangerousGrids) and len(routes[x][y]) and len(routes[x][y]) < len(EscapeRoute):
                #  ^~~~~~~安全区域                ^~~~~可达             ^~~~~~~~~~~~~~~~~~~~~~~路径更短
                EscapeRoute = routes[x][y]
    if len(EscapeRoute) == 255:
        return []
    else:
        return EscapeRoute


# 行走接口
def safeGoTo(targets : List[tuple], routes : List[List[List[tuple]]], playerPosition : tuple, dangerousGrids: List[List]) -> List[ActionReq]:
    targetPath = [tuple() for i in range(999)]
    for target in targets:
        if insideGrids(target) == False:
            logger.error("safeGoTo(): Target out of range, Check your arguements!")
        else:
            # 量词 any 为存在量词
            # if not any(step in dangerousGrids for step in routes[target[0]][target[1]]) and len(routes[target[0]][target[1]]) and len(routes[target[0]][target[1]]) < len(targetPath):
            if all(step not in dangerousGrids for step in routes[target[0]][target[1]]) and len(routes[target[0]][target[1]]) and len(routes[target[0]][target[1]]) < len(targetPath):
                targetPath = routes[target[0]][target[1]]
    if len(targetPath) == 999:
        print("safeGoTo() : No route to go!")
        return []
    else:
        print("safeGoto() : Heading to " + str(targetPath[-1]))
        print("safeGoTo() : Route: " + str(routes[target[0]][target[1]]))
        return routeToActionReq(targetPath) + [ActionReq(gContext["playerID"], ActionType.SILENT)]
        

# 不要调用
def routeToActionReq(route: List[tuple]) -> List[ActionReq]:
    steps = []
    for i in range(1, len(route)):
        diff = (route[i][0] - route[i-1][0], route[i][1] - route[i-1][1])
        if diff[0] == 0 and diff[1] == 1:
            steps.append(ActionType.MOVE_RIGHT)
        elif diff[0] == 0 and diff[1] == -1:
            steps.append(ActionType.MOVE_LEFT)
        elif diff[0] == 1 and diff[1] == 0:
            steps.append(ActionType.MOVE_DOWN)
        elif diff[0] == -1 and diff[1] == 0:
            steps.append(ActionType.MOVE_UP)
        else:
            logger.error("Wrong step!")
    if len(steps) == 0:
        return []
    return [ActionReq(gContext["playerID"], x) for x in steps]


# 判定出界
def insideGrids(grid : tuple) -> bool:
    return grid[0] >= 0 and grid[0] < MapEdgeLength and grid[1] >= 0 and grid[1] < MapEdgeLength


# 尽量让玩家冲向bot避免被炸弹堵死
def SeekEnemyAndAttack(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
              playerPosition: tuple, enemyPosition: dict, dangerousGrids: List[tuple]) -> List[ActionReq]:
    targets = []
    nowZoneOfBomb = FindZoneOfBomb(parsedMap, playerPosition)
    for enemy in enemyPosition.keys():
        if enemyPosition[enemy] in nowZoneOfBomb:
            return PlaceBomb(parsedMap, routes, playerPosition, dangerousGrids)
        else:
            if len(routes[enemyPosition[enemy][0]][enemyPosition[enemy][1]]) and len(routes[enemyPosition[enemy][0]][enemyPosition[enemy][1]]) <= attackDistance:
                targets.append(enemyPosition[enemy])
    print("seekEnemy(): calling safeGoTo()")
    return safeGoTo(targets, routes, playerPosition, dangerousGrids)


# 追击敌人
def ChaseEnemyAndAttack(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
              playerPosition: tuple, enemyPosition: dict, dangerousGrids: List[tuple]) -> List[ActionReq]:
    targets = []
    nowZoneOfBomb = FindZoneOfBomb(parsedMap, playerPosition)
    for enemy in enemyPosition.keys():
        if enemyPosition[enemy] in nowZoneOfBomb:
            return PlaceBomb(parsedMap, routes, playerPosition, dangerousGrids)
        else:
            if len(routes[enemyPosition[enemy][0]][enemyPosition[enemy][1]]) and len(routes[enemyPosition[enemy][0]][enemyPosition[enemy][1]]) <= ChaseDistance:
                targets.append(enemyPosition[enemy])
    print("seekEnemy(): calling safeGoTo()")
    return safeGoTo(targets, routes, playerPosition, dangerousGrids)



def Play(Map: List[List[Map]]) -> List[ActionReq]:
    parsedMap, routes, playerPosition, enemyPosition, isInDangerousZone, desperate, dangerousGrids = ParseMap(resp.data.map)
    actionReqList = [] 

    if isInDangerousZone:
        tmpReqList =  EscapeToSafeZone(parsedMap, routes, playerPosition, dangerousGrids)
        actionReqList += tmpReqList 
    if len(actionReqList) > maxSpeed:
        return actionReqList[0:maxSpeed]
    
    tmpReqList = SeekEnemyAndAttack(parsedMap, routes, playerPosition, enemyPosition, dangerousGrids)
    actionReqList += tmpReqList
    if len(actionReqList) > maxSpeed:
        return actionReqList[0:maxSpeed]
    
    tmpReqList = GoToItem(parsedMap, routes, playerPosition, dangerousGrids)
    actionReqList += tmpReqList
    if len(actionReqList) > maxSpeed:
        return actionReqList[0:maxSpeed]
    
    tmpReqList = GoToRemovableBlockAndPlaceBomb(parsedMap, routes, playerPosition, dangerousGrids)
    actionReqList += tmpReqList
    if len(actionReqList) > maxSpeed:
        return actionReqList[0:maxSpeed]
    
    # 地图上没有可炸的砖块时追击敌人
    tmpReqList = ChaseEnemyAndAttack(parsedMap, routes, playerPosition, enemyPosition, dangerousGrids)
    actionReqList += tmpReqList
    if len(actionReqList) > maxSpeed:
        return actionReqList[0:maxSpeed]
    
    return actionReqList
    # 切片是为了防止server报警告导致画面跳动过于剧烈
# def Desperate(parsedMap: List[List[Map]], routes: List[List[List[tuple]]]),)


# def DesperateEscape(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],


def FindZoneOfBomb(parsedMap: List[List[Map]], Bomb : tuple) -> List[tuple]:
    def MeetBunker(parsedMap: List[List[Map]], gridToCheck : tuple):
        for obj in parsedMap[gridToCheck[0]][gridToCheck[1]].objs:
            if obj.type == ObjType.Block:
                return True
        return False 
    directions = [[-1,0],[0,1],[1,0],[0,-1]]
    ThisBombZone = []
    bomb = None
    for obj in parsedMap[Bomb[0]][Bomb[1]].objs:
        if obj.type == ObjType.Bomb:
            bomb = obj
    x = Bomb[0]
    y = Bomb[1]
    for dis in range(obj.property.bomb_range + 1):
        for direction in directions:
            gridToCheck = (x + direction[0] * dis, y + direction[1] * dis)
            if insideGrids(gridToCheck):
                if(MeetBunker(parsedMap, gridToCheck)):
                    continue
                else:
                    ThisBombZone.append(gridToCheck)
    return ThisBombZone


# 格局上所有危险位置，以及我是否在危险位置上
def AnalyseDanger(parsedMap: List[List[Map]], playerPosition: tuple, routes: List[List[List[tuple]]]) -> (bool, bool, List[tuple]):
    inDangerZone = False
    desperate = False
    dangerousGrids = [] 
    
    for x in range(MapEdgeLength):
        for y in range(MapEdgeLength):
            for obj in parsedMap[x][y].objs:
                if obj.type == ObjType.Bomb:
                    dangerousGrids += FindZoneOfBomb(parsedMap, (x,y))
    if playerPosition in dangerousGrids:
        inDangerZone = True
    EscapeRoute = ChooseEscapeRoute(routes, playerPosition, dangerousGrids)
    if len(EscapeRoute) == 0:
        desperate = True
        print("AnalyseDanger(): I am so Desperate!")
    return inDangerZone, desperate, dangerousGrids


def ParseMap(map:List[Map]) -> (List[List[Map]], List[List[List[tuple]]], tuple, dict, bool, bool, List[tuple]):
    global maxSpeed
    global hasGloves
    parsedMap = [[Map() for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    paths = [[[] for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    accessableNow = [[1 for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    myPosition = None
    enemyPosition = {}
    for grid in map:
        parsedMap[grid.x][grid.y] = grid
        for obj in grid.objs:
            if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"] and myPosition is None:
                myPosition = (grid.x, grid.y)
                maxSpeed = obj.property.speed
                hasGloves = obj.property.has_gloves
            if obj.type == ObjType.Player and obj.property.player_id != gContext["playerID"]:
                enemyPosition[obj.property.player_id] = (grid.x, grid.y)
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
    InDangerousZone, Desperate, dangerousGrids = AnalyseDanger(parsedMap, myPosition, paths)
    return parsedMap, paths, myPosition, enemyPosition, InDangerousZone, Desperate, dangerousGrids


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
        requests = Play(Map)
        client.send(PacketReq(PacketType.ActionReq, requests))
        resp = client.recv()    
        if resp.type == PacketType.GameOver:
            gContext["gameOverFlag"] = True
            for score in resp.data.scores:
                if score["player_id"] == gContext["playerID"]: 
                    gContext["result"] = score["score"]
            logger.info(f"game over: {gContext['gameOverFlag']}, my score: {gContext['result']}")
            break