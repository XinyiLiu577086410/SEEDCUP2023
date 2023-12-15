from typing import List
from req import *
from base import *
from req import *
from resp import *
from config import config
from logger import logger
import math
import json
import socket
import sys
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
# from pathfinding.finder.dijkstra import DijkstraFinder
from pathfinding.finder.a_star import AStarFinder

BaseAttackDistance = 8
AttackDistance = BaseAttackDistance # 攻击距离, 应避免太大
ChaseDistance = 999
# MaxSpeed = 2 # 移动速度
HasGloves = False # 能否推动炸弹
BombInfo = []
CurrentTurn = 0
enemyPosition = {}
gContext = {
    "playerID": -1,
    "gameOverFlag": False,
    "result": 0,
    "gameBeginFlag": False,
}
MapEdgeLength = 19
Player = None
Invincible = False
Shielded = False
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
             playerPosition: tuple, bannedGrids: List[tuple]) -> List[ActionReq]:
    print("GoToItem()")
    # NO_POTION = 0
    # BOMB_RANGE = 1
    # BOMB_NUM = 2
    # HP = 3
    # INVINCIBLE = 4
    # SHIELD = 5
    # SPEED = 6
    # GLOVES = 7
    global HasGloves
    global Player
    global aroundInvincibleEnermy
    targets = [[] for i in range(8)]
    for i in range(MapEdgeLength):
        for j in range(MapEdgeLength):
            for obj in parsedMap[i][j].objs:
                if obj.type == ObjType.Item:
                    targets[obj.property.item_type].append((i,j))

    print("GoToItem(): Weighting")
    orders = [6, 4, 3, 7, 5, 1, 2 ]
    if Player.property.hp == 1:
        orders = [3, 6, 4, 7, 5, 1, 2 ]
    if Player.property.bomb_max_num > 6:
        orders = [6, 4, 7, 5, 1, 3, 2]
    for order in orders:
        if len(targets[order]):
            if order == 7 and HasGloves:
                continue
            if order == 3 and Player.property.hp > 1:
                continue
            if order == 4:
                decision = safeGoTo(targets[order], routes, playerPosition, bannedGrids)
                if len(decision):
                    return decision
            else:           
                decision = safeGoTo(targets[order], routes, playerPosition, bannedGrids+aroundInvincibleEnermy)
                if len(decision):
                    return decision
    return []


def NextToRemovableBlock(parsedMap: List[List[Map]], playerPosition: tuple, dangerousGrids: list[tuple]) -> bool:
    print("NextToRemovableBlock()")
    directions = [[-1,0],[0,1],[1,0],[0,-1]]
    print("NextToRemovableBlock(): ", end="")
    print(playerPosition)
    for dir in directions:
        GridToCheck = (playerPosition[0]+dir[0], playerPosition[1]+dir[1])
        if insideGrids(GridToCheck):
            for obj in parsedMap[GridToCheck[0]][GridToCheck[1]].objs:
                if obj.type == ObjType.Block and obj.property.removable == True and not GridToCheck in dangerousGrids:
                    print("NextToRemovableBlock(): True")
                    return True
    return False


def GoToRemovableBlockAndPlaceBomb(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
                            playerPosition: tuple, bannedGrids: List[tuple]) -> List[ActionReq]:
    print("GoToRemovableBlockAndPlaceBomb()")
    targets = []
    directions = [[-1,0],[0,1],[1,0],[0,-1]]
    if NextToRemovableBlock(parsedMap, playerPosition, bannedGrids):
        return PlaceBomb(parsedMap, routes, playerPosition, bannedGrids)
    for i in range(MapEdgeLength):
        for j in range(MapEdgeLength):    
            for obj in parsedMap[i][j].objs:
                if obj.type == ObjType.Block and obj.property.removable == True and not (i,j) in bannedGrids:
                    if insideGrids((i+1,j)):
                        targets.append((i+1,j))
                    if insideGrids((i-1,j)):
                        targets.append((i-1,j))
                    if insideGrids((i,j+1)):
                        targets.append((i,j+1))
                    if insideGrids((i,j-1)):
                        targets.append((i,j-1))
    targets = list(set(targets))
    print("GoToRemoveableBlock(): calling safeGoTo()")
    return safeGoTo(targets, routes, playerPosition, bannedGrids)

PlaceBombCallFlag = False
def PlaceBomb(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
                playerPosition: tuple, bannedGrids : List[tuple]) -> List[ActionReq]:
    print("PlaceBomb()")
    global PlaceBombCallFlag
    # 检查当前位置能否放下炸弹
    for obj in parsedMap[playerPosition[0]][playerPosition[1]].objs:
        if obj.type == ObjType.Bomb:
            return []
    # 检查
    # a 获取炸弹范围
    bombRange = 0
    for obj in parsedMap[playerPosition[0]][playerPosition[1]].objs:
        if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
            bombRange = obj.property.bomb_range    
    # b 检查放完炸弹后的安全性
    changedMap = parsedMap.copy()
    changedMap[playerPosition[0]][playerPosition[1]].objs.append(Obj(ObjType.Bomb, Bomb(999, bombRange, gContext["playerID"], BombStatus.BOMB_SILENT)))

    # 不要冒险
    safeRoutes = routes.copy()
    for i in range(MapEdgeLength):
        for j in range(MapEdgeLength):
           for step in routes[i][j]:
               if step in bannedGrids:
                   safeRoutes[i][j] = [] 
                   break
    PlaceBombCallFlag = True
    _, desperate, changedDangerousGrids = AnalyseDanger(changedMap, playerPosition, safeRoutes)
    if desperate:
        return []
    else:
        print("PlaceBomb(): Bomb placed at " + str(playerPosition))
        return [ActionReq(gContext["playerID"], ActionType.PLACED)] + EscapeToSafeZone(changedMap, safeRoutes, playerPosition, changedDangerousGrids)
    

# 仅当位于危险区域时调用
# 会冒险通过危险区域逃生
def EscapeToSafeZone(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
                    playerPosition: tuple, bannedGrids: List[List[tuple]]) -> List[ActionReq]: 
    print("EscapeToSafeZone()")
    EscapeRoute = ChooseEscapeRoute(routes, playerPosition, bannedGrids)
    if len(EscapeRoute) == 0:
        print("EscapeToSafeZone(): No route to escape")
        return DesperateEscape() # 无路可走
    else:
        pass        
        print("EscapeToSafeZone(): Going to " + str(EscapeRoute[-1]))
    return routeToActionReq(EscapeRoute) + [ActionReq(gContext["playerID"], ActionType.SILENT) for i in range(10)] # 避免后续的冒险行为


def ChooseEscapeRoute(routes : List[List[List[tuple]]], 
                        playerPosition : tuple, bannedGrids : List[tuple]) -> List[tuple]:
    print("ChooseEscapeRoute()")
    EscapeRoute = [tuple() for i in range(255)]
    for x in range(MapEdgeLength):
        for y in range(MapEdgeLength):
            # 注意 route 为空的情况表示不可达
            if not ((x,y) in bannedGrids) and len(routes[x][y]) and len(routes[x][y]) < len(EscapeRoute):
                #  ^~~~~~~安全区域                ^~~~~可达             ^~~~~~~~~~~~~~~~~~~~~~~路径更短
                EscapeRoute = routes[x][y]
    if len(EscapeRoute) == 255:
        return []
    else:
        return EscapeRoute


# 行走接口
def safeGoTo(targets : List[tuple], routes : List[List[List[tuple]]], playerPosition : tuple, bannedGrids: List[List]) -> List[ActionReq]:
    print("safeGoTo()")
    targetPath = [tuple() for i in range(999)]
    for target in targets:
        if insideGrids(target) == False:
            logger.error("safeGoTo(): Target out of range, Check your arguements!")
        else:
            # 量词 any 为存在量词
            # if not any(step in dangerousGrids for step in routes[target[0]][target[1]]) and len(routes[target[0]][target[1]]) and len(routes[target[0]][target[1]]) < len(targetPath):
            if all(step not in bannedGrids for step in routes[target[0]][target[1]]) and len(routes[target[0]][target[1]]) and len(routes[target[0]][target[1]]) < len(targetPath):
                targetPath = routes[target[0]][target[1]]
    if len(targetPath) == 999:
        if len(targets):
            pass
            print(f"safeGoTo() : Target exists but no route to go: {len(targets)}")
        return []
    else:
        print("safeGoto() : Heading to " + str(targetPath[-1]))
        print("safeGoTo() : RouteLen: " + str(len(routes[targetPath[-1][0]][targetPath[-1][1]])))
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


def reverseOperation(ActionReq : ActionReq) -> ActionReq:
    if ActionReq.actionType == ActionType.MOVE_UP:
        return ActionReq(gContext["playerID"], ActionType.MOVE_DOWN)
    elif ActionReq.actionType == ActionType.MOVE_DOWN:
        return ActionReq(gContext["playerID"], ActionType.MOVE_UP)
    elif ActionReq.actionType == ActionType.MOVE_LEFT:
        return ActionReq(gContext["playerID"], ActionType.MOVE_RIGHT)
    elif ActionReq.actionType == ActionType.MOVE_RIGHT:
        return ActionReq(gContext["playerID"], ActionType.MOVE_LEFT)
    else:
        return ActionReq(gContext["playerID"], ActionType.SILENT)

# 尽量让玩家冲向bot避免被炸弹堵死
def SeekEnemyAndAttack(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
              playerPosition: tuple, enemyPosition: dict, bannedGrids: List[tuple]) -> List[ActionReq]:
    print("SeekEnemyAndAttack()")
    global AttackDistance
    global Invincible
    global Shielded
    global HasGloves
    targets = []
    nowZoneOfBomb = FindZoneOfBomb(parsedMap, playerPosition)
    for enemy in enemyPosition.keys():
        if enemyPosition[enemy] in nowZoneOfBomb and not Invincible:
            push = []
            quit = []
            if HasGloves:
                if enemyPosition[enemy][0] > playerPosition[0]:
                    if insideGrids((playerPosition[0]-1, playerPosition[1])) and not MeetBunker(parsedMap, (playerPosition[0]-1, playerPosition[1])):
                        push += [ActionReq(gContext["playerID"], ActionType.MOVE_UP) , ActionReq(gContext["playerID"], ActionType.MOVE_DOWN)]
                elif enemyPosition[enemy][0] < playerPosition[0]:
                    if insideGrids((playerPosition[0]+1, playerPosition[1])) and not MeetBunker(parsedMap, (playerPosition[0]+1, playerPosition[1])):
                        push += [ActionReq(gContext["playerID"], ActionType.MOVE_DOWN) , ActionReq(gContext["playerID"], ActionType.MOVE_UP)]
                elif enemyPosition[enemy][1] > playerPosition[1]:
                    if insideGrids((playerPosition[0], playerPosition[1]-1)) and not MeetBunker(parsedMap, (playerPosition[0], playerPosition[1]-1)):
                        push += [ActionReq(gContext["playerID"], ActionType.MOVE_LEFT) , ActionReq(gContext["playerID"], ActionType.MOVE_RIGHT)]
                elif enemyPosition[enemy][1] < playerPosition[1]:
                    if insideGrids((playerPosition[0], playerPosition[1]+1)) and not MeetBunker(parsedMap, (playerPosition[0], playerPosition[1]+1)):
                        push += [ActionReq(gContext["playerID"], ActionType.MOVE_RIGHT) , ActionReq(gContext["playerID"], ActionType.MOVE_LEFT)]
            return PlaceBomb(parsedMap, routes, playerPosition, bannedGrids) + push + quit # push the bomb towards enemy
        else:
            if Invincible:
                if len(routes[enemyPosition[enemy][0]][enemyPosition[enemy][1]]):
                    for obj in parsedMap[enemyPosition[enemy][0]][enemyPosition[enemy][1]].objs:
                        if obj.type == ObjType.Player and obj.property.player_id != gContext["playerID"] and obj.property.shield_time < 3:
                            if not obj.property.invincible_time:
                                targets.append(enemyPosition[enemy])
            else:
                if len(routes[enemyPosition[enemy][0]][enemyPosition[enemy][1]]) \
                    and len(routes[enemyPosition[enemy][0]][enemyPosition[enemy][1]]) <= AttackDistance:
                    for obj in parsedMap[enemyPosition[enemy][0]][enemyPosition[enemy][1]].objs:
                        if obj.type == ObjType.Player and obj.property.player_id != gContext["playerID"] and obj.property.shield_time < 3:
                            if not obj.property.invincible_time:
                                targets.append(enemyPosition[enemy])


    print("seekEnemy(): calling safeGoTo()")
    if Invincible:
        steps = safeGoTo(targets, routes, playerPosition, [])
        print("seekEnemy(): now invincible")
        if len(steps):
            print (steps + [steps[-1],steps[-1],steps[-1],])
            return steps + [steps[-1],steps[-1],steps[-1],]
        else:
            return steps
    if Shielded:
        return safeGoTo(targets, routes, playerPosition, [])
    else:
        return safeGoTo(targets, routes, playerPosition, bannedGrids)


# 追击敌人
def ChaseEnemyAndAttack(parsedMap: List[List[Map]], routes: List[List[List[tuple]]],
              playerPosition: tuple, enemyPosition: dict, bannedGrids: List[tuple]) -> List[ActionReq]:
    print("ChaseEnemyAndAttack()")
    # must use global keyword here
    global AttackDistance
    global ChaseDistance
    tmp = AttackDistance
    AttackDistance = ChaseDistance
    ret = SeekEnemyAndAttack(parsedMap, routes, playerPosition, enemyPosition, bannedGrids)
    AttackDistance = tmp
    return ret

# def calcPlayerPosition(actionReqList : list[ActionReq], playerPosition : tuple[int, int]):
#     changedPosition = playerPosition

#     for action in actionReqList:
#         if action.actionType == ActionType.MOVE_LEFT:
#             changedPosition = (changedPosition[0],changedPosition[1]-1)
#         elif action.actionType == ActionType.MOVE_RIGHT:
#             changedPosition = (changedPosition[0],changedPosition[1]+1)
#         elif action.actionType == ActionType.MOVE_UP:
#             changedPosition = (changedPosition[0]-1,changedPosition[1])
#         elif action.actionType == ActionType.MOVE_DOWN:
#             changedPosition = (changedPosition[0]+1,changedPosition[1])
#     # if changedPosition == playerPosition:
#     #     print("calcPlayerPosition: no changes")
#     return changedPosition


vis = []

def dfs(accessableNow, start, step):
    global vis
    if step == 0:
        return []
    res = [start]
    vis.append(start)
    directions = [[-1,0],[0,1],[1,0],[0,-1]]
    for direction in directions:
        gridToCheck = (start[0] + direction[0], start[1] + direction[1])
        if insideGrids(gridToCheck) and accessableNow[gridToCheck[0]][gridToCheck[1]] and gridToCheck not in vis:
            res += dfs(accessableNow, gridToCheck, step-1)
    return res
        
aroundEnermy = []
aroundInvincibleEnermy = []
def KeepAwayFromInvinciblePlayer(parsedMap: List[List[Map]]) :
    print("KeepAwayFromInvinciblePlayer()")
    global enemyPosition
    global aroundEnermy
    global aroundInvincibleEnermy
    killerPosition = {}
    aroundEnermy.clear()
    aroundInvincibleEnermy.clear()
    for enemy in enemyPosition.values():
        aroundEnermy += dfs(accessableNow, enemy, 6)
        # for obj in parsedMap[enemy[0]][enemy[1]].objs:
        #     if obj.type == ObjType.Player and obj.property.player_id != gContext["playerID"]:
        #         if obj.property.invincible_time > 0:
        #             killerPosition[enemy] = obj.property.speed
    for killer in killerPosition.keys():
        aroundInvincibleEnermy += dfs(accessableNow, killer, killerPosition[killer]*3)
        # for i in range(killerPosition[killer]):
        #     for j in range(killerPosition[killer]):
        #         if insideGrids((killer[0]+i,killer[1]+j)):
        #             dangerousGrids.append((killer[0]+i,killer[1]+j))
        #         if insideGrids((killer[0]-i,killer[1]-j)):
        #             dangerousGrids.append((killer[0]-i,killer[1]-j))
        #         if insideGrids((killer[0]-i,killer[1]+j)):
        #             dangerousGrids.append((killer[0]-i,killer[1]+j))
        #         if insideGrids((killer[0]+i,killer[1]-j)):
        #             dangerousGrids.append((killer[0]+i,killer[1]-j))

# def Play(resp : ActionResp) -> List[ActionReq]:
#     parsedMap, routes, playerPosition, enemyPosition, isInDangerousZone, desperate, dangerousGrids = ParseMap(resp.data.map)
#     actionReqList = [] 
#     global maxSpeed
#     global Invincible
#     global AttackDistance
#     if Invincible:
#         AttackDistance = 999

#     if isInDangerousZone:
#         tmpReqList = EscapeToSafeZone(parsedMap, routes, playerPosition, dangerousGrids+aroundInvincibleEnermy+aroundEnermy)
#         actionReqList += tmpReqList 
#         tmpReqList = EscapeToSafeZone(parsedMap, routes, playerPosition, dangerousGrids+aroundInvincibleEnermy)
#         actionReqList += tmpReqList 
#     if len(actionReqList) > MaxSpeed:
#         return actionReqList[0:MaxSpeed]
    
    

#     tmpReqList = GoToItem(parsedMap, routes, playerPosition, dangerousGrids+aroundInvincibleEnermy)
#     actionReqList += tmpReqList
#     if len(actionReqList) > MaxSpeed:
#         return actionReqList[0:MaxSpeed]
    

#     tmpReqList = SeekEnemyAndAttack(parsedMap, routes, playerPosition, enemyPosition, dangerousGrids+aroundInvincibleEnermy)
#     AttackDistance = 8
#     actionReqList += tmpReqList
#     if len(actionReqList) > MaxSpeed:
#         return actionReqList[0:MaxSpeed]
    
    
#     tmpReqList = GoToRemovableBlockAndPlaceBomb(parsedMap, routes, playerPosition, dangerousGrids+aroundInvincibleEnermy+aroundEnermy)
#     actionReqList += tmpReqList
#     if len(actionReqList) > MaxSpeed:
#         return actionReqList[0:MaxSpeed]
    
    
#     tmpReqList = ChaseEnemyAndAttack(parsedMap, routes, playerPosition, enemyPosition, dangerousGrids+aroundInvincibleEnermy)
#     actionReqList += tmpReqList
#     if len(actionReqList) > MaxSpeed:
#         return actionReqList[0:MaxSpeed]
    
    
#     return actionReqList
#     # 切片是为了防止server报警告导致画面跳动过于剧烈


# def DesperateEscape():
#     return [ActionReq(gContext["playerID"], ActionType.MOVE_UP), ActionReq(gContext["playerID"], ActionType.MOVE_UP), 
#             ActionReq(gContext["playerID"], ActionType.MOVE_LEFT), ActionReq(gContext["playerID"], ActionType.MOVE_RIGHT)]


# def MeetBunker(parsedMap: List[List[Map]], gridToCheck : tuple):
#     for obj in parsedMap[gridToCheck[0]][gridToCheck[1]].objs:
#         if obj.type == ObjType.Block:
#             return True
#     return False 


# def FindZoneOfBomb(parsedMap: List[List[Map]], Bomb : tuple[int, int]) -> List[tuple]:
#     print("ZoneOfBomb()")
#     directions = [[-1,0],[0,1],[1,0],[0,-1]]
#     ThisBombZone = []
#     bomb_range = -1
#     for obj in parsedMap[Bomb[0]][Bomb[1]].objs:
#         if obj.type == ObjType.Bomb:
#             bomb_range = obj.property.bomb_range

#     # 如果这里没有炸弹，则假设放下一个炸弹
#     if bomb_range == -1:
#         for x in range(MapEdgeLength):
#             for y in range(MapEdgeLength):
#                 for obj in parsedMap[x][y].objs:
#                     if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
#                         bomb_range = obj.property.bomb_range

#     x = Bomb[0]
#     y = Bomb[1]
#     for dis in range(bomb_range + 1):
#         newDirections = directions.copy()
#         for direction in directions:
#             gridToCheck = (x + direction[0] * dis, y + direction[1] * dis)
#             if insideGrids(gridToCheck):
#                 if(MeetBunker(parsedMap, gridToCheck)):
#                     newDirections.remove(direction)
#                 ThisBombZone.append(gridToCheck)
#             else: 
#                 newDirections.remove(direction)
#         directions = newDirections
#     ThisBombZone = list(set(ThisBombZone))
#     print(f"ZoneOfBomb(): Bomb: {Bomb} Zone: {len(ThisBombZone)}")
#     return ThisBombZone


# # 格局上所有危险位置，以及我是否在危险位置上
# def AnalyseDanger(parsedMap: List[List[Map]], playerPosition: tuple, routes: List[List[List[tuple]]]) -> (bool, bool, List[tuple]):

#     print("AnalyseDanger()")
#     global PlaceBombCallFlag
#     inDangerZone = False
#     desperate = False
#     dangerousGrids_ = [] 
#     for x in range(MapEdgeLength):
#         for y in range(MapEdgeLength):
#             for obj in parsedMap[x][y].objs:
#                 if obj.type == ObjType.Bomb:
#                     dangerousGrids_ += FindZoneOfBomb(parsedMap, (x,y))
    
#     # take invincible enemys into consideration
#     if not PlaceBombCallFlag:
#         KeepAwayFromInvinciblePlayer(parsedMap)
#     dangerousGrids_ = list(set(dangerousGrids_))
#     if playerPosition in (dangerousGrids_ + aroundInvincibleEnermy):
#         inDangerZone = True
#     EscapeRoute = ChooseEscapeRoute(routes, playerPosition, dangerousGrids_)
#     if len(EscapeRoute) == 0:
#         desperate = True
#         print("AnalyseDanger(): I am so Desperate!")
#     PlaceBombCallFlag = False
#     return inDangerZone, desperate, dangerousGrids_

AttackedLastTurn = False

def Play(resp : ActionResp) -> List[ActionReq]:
    parsedMap, routes, playerPosition, enemyPosition, isInDangerousZone, desperate, dangerousGrids = ParseMap(resp.data.map)
    actionReqList = [] 
    # global maxSpeed 
    global Invincible
    global AttackDistance
    global Shielded
    global Player
    global BaseAttackDistance
    global AttackedLastTurn
    global AttackDistance

    if AttackedLastTurn:
        AttackDistance += 4*(Player.property.bomb_range + Player.property.speed) + BaseAttackDistance
    else:
        AttackDistance = Player.property.bomb_range + Player.property.speed + BaseAttackDistance


    if Invincible or (Shielded and Player.property.shield_time > 5) or AttackedLastTurn:
        AttackDistance = 999
        actionReqList = SeekEnemyAndAttack(parsedMap, routes, playerPosition, enemyPosition, dangerousGrids+aroundInvincibleEnermy)
        AttackDistance = 8
        if actionReqList != []:
            AttackedLastTurn = True
            return actionReqList
        else:
            AttackedLastTurn = False
            pass


    if isInDangerousZone:
        actionReqList = EscapeToSafeZone(parsedMap, routes, playerPosition, dangerousGrids+aroundInvincibleEnermy+aroundEnermy)
        if actionReqList != []:
            return actionReqList
        else :
            actionReqList = EscapeToSafeZone(parsedMap, routes, playerPosition, dangerousGrids+aroundInvincibleEnermy)
            if actionReqList != []:
                return actionReqList
            else:
                pass
    
    else: 
        actionReqList = GoToItem(parsedMap, routes, playerPosition, dangerousGrids+aroundInvincibleEnermy)
        if actionReqList != []:
            return actionReqList
        else :
            pass

        actionReqList = SeekEnemyAndAttack(parsedMap, routes, playerPosition, enemyPosition, dangerousGrids+aroundInvincibleEnermy)
        AttackDistance = 8
        if actionReqList != []:
            AttackedLastTurn = True
            return actionReqList
        else:
            AttackedLastTurn = False
            pass
            
        actionReqList = GoToRemovableBlockAndPlaceBomb(parsedMap, routes, playerPosition, dangerousGrids+aroundInvincibleEnermy+aroundEnermy)
        
        if actionReqList != []:
            return actionReqList
        
        
        actionReqList = ChaseEnemyAndAttack(parsedMap, routes, playerPosition, enemyPosition, dangerousGrids+aroundInvincibleEnermy)
        if actionReqList != []:
            AttackedLastTurn = True
            return actionReqList
        else:
            AttackedLastTurn = False
            return []
        

def DesperateEscape():
    return [ActionReq(gContext["playerID"], ActionType.MOVE_UP), ActionReq(gContext["playerID"], ActionType.MOVE_UP), 
            ActionReq(gContext["playerID"], ActionType.MOVE_LEFT), ActionReq(gContext["playerID"], ActionType.MOVE_RIGHT)]


def MeetBunker(parsedMap: List[List[Map]], gridToCheck : tuple):
    for obj in parsedMap[gridToCheck[0]][gridToCheck[1]].objs:
        if obj.type == ObjType.Block:
            return True
    return False 


def FindZoneOfBomb(parsedMap: List[List[Map]], Bomb : tuple[int, int]) -> List[tuple]:
    print("ZoneOfBomb()")
    directions = [[-1,0],[0,1],[1,0],[0,-1]]
    ThisBombZone = []
    bomb_range = -1
    for obj in parsedMap[Bomb[0]][Bomb[1]].objs:
        if obj.type == ObjType.Bomb:
            bomb_range = obj.property.bomb_range

    # 如果这里没有炸弹，则假设放下一个炸弹
    if bomb_range == -1:
        for x in range(MapEdgeLength):
            for y in range(MapEdgeLength):
                for obj in parsedMap[x][y].objs:
                    if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
                        bomb_range = obj.property.bomb_range

    x = Bomb[0]
    y = Bomb[1]
    for dis in range(bomb_range + 1):
        newDirections = directions.copy()
        for direction in directions:
            gridToCheck = (x + direction[0] * dis, y + direction[1] * dis)
            if insideGrids(gridToCheck):
                if(MeetBunker(parsedMap, gridToCheck)):
                    newDirections.remove(direction)
                ThisBombZone.append(gridToCheck)
            else: 
                newDirections.remove(direction)
        directions = newDirections
    ThisBombZone = list(set(ThisBombZone))
    print(f"ZoneOfBomb(): Bomb: {Bomb} Zone: {len(ThisBombZone)}")
    return ThisBombZone


# 格局上所有危险位置，以及我是否在危险位置上
def AnalyseDanger(parsedMap: List[List[Map]], playerPosition: tuple, routes: List[List[List[tuple]]]) -> (bool, bool, List[tuple]):

    print("AnalyseDanger()")
    global PlaceBombCallFlag
    inDangerZone = False
    desperate = False
    dangerousGrids_ = [] 
    for x in range(MapEdgeLength):
        for y in range(MapEdgeLength):
            for obj in parsedMap[x][y].objs:
                if obj.type == ObjType.Bomb:
                    dangerousGrids_ += FindZoneOfBomb(parsedMap, (x,y))

    for obj in parsedMap[playerPosition[0]][playerPosition[1]].objs:
        if obj.type == ObjType.Player and obj.property.player_id != gContext["playerID"]:
            dangerousGrids_ += playerPosition

    # take invincible enemys into consideration
    if not PlaceBombCallFlag:
        KeepAwayFromInvinciblePlayer(parsedMap)
    dangerousGrids_ = list(set(dangerousGrids_))
    if playerPosition in (dangerousGrids_ + aroundInvincibleEnermy):
        inDangerZone = True
    EscapeRoute = ChooseEscapeRoute(routes, playerPosition, dangerousGrids_)
    if len(EscapeRoute) == 0:
        desperate = True
        print("AnalyseDanger(): I am so Desperate!")
    PlaceBombCallFlag = False
    return inDangerZone, desperate, dangerousGrids_



accessableNow = []
def ParseMap(map:List[Map]) -> (List[List[Map]], List[List[List[tuple]]], tuple, dict, bool, bool, List[tuple]):
    global MaxSpeed
    global HasGloves
    global CurrentTurn
    global BombInfo
    global enemyPosition
    global MapEdgeLength
    global Invincible
    global Shielded
    global accessableNow
    global Player
    Invincible = False
    Shielded = False
    parsedMap = [[Map() for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    paths = [[[] for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    accessableNow = [[1 for i in range(MapEdgeLength)] for j in range(MapEdgeLength)]
    CurrentTurn = resp.data.round
    myPosition = None
    enemyPosition = {}
    newBombInfo = []
    for grid in map:
        parsedMap[grid.x][grid.y] = grid
        for obj in grid.objs:
            if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"] and myPosition is None:
                Player = obj
                myPosition = (grid.x, grid.y)
                MaxSpeed = obj.property.speed
                HasGloves = obj.property.has_gloves
            if obj.type == ObjType.Player and obj.property.player_id != gContext["playerID"]:
                enemyPosition[obj.property.player_id] = (grid.x, grid.y)
            if obj.type == ObjType.Block or obj.type == ObjType.Bomb:
                accessableNow[grid.x][grid.y] = 0 
            if obj.type == ObjType.Bomb:
                if not (obj.property.bomb_id in list(data[0] for data in BombInfo)):
                    newBombInfo.append((obj.property.bomb_id, grid.x, grid.y, CurrentTurn))
                else:
                    oldBombInfo = list(filter(lambda data : data[0] == obj.property.bomb_id, BombInfo))
                    newBombInfo.append((obj.property.bomb_id, grid.x, grid.y, oldBombInfo[0][3]))
    if Player.property.invincible_time > 0:
        Invincible = True
    if Player.property.shield_time > 0:
        Shielded = True
    newBombInfo.sort(key=lambda x:x[0])
    BombInfo = newBombInfo
    print(f"BombList: {BombInfo}")
    # if no position detected, already dead, exit
    if myPosition == None:
        exit(0)
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

    
    InDangerousZone, Desperate, dangerousGrids_local = AnalyseDanger(parsedMap, myPosition, paths)

    return parsedMap, paths, myPosition, enemyPosition, InDangerousZone, Desperate, dangerousGrids_local


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
    MapEdgeLength = int(math.sqrt(len(resp.data.map)))
    print(f"detected map edge length {MapEdgeLength}")
    while(not gContext["gameOverFlag"]):
        requests = Play(resp)
        client.send(PacketReq(PacketType.ActionReq, requests))
        resp = client.recv()    
        if resp.type == PacketType.GameOver:
            gContext["gameOverFlag"] = True
            for score in resp.data.scores:
                if score["player_id"] == gContext["playerID"]: 
                    gContext["result"] = score["score"]
            logger.info(f"game over: {gContext['gameOverFlag']}, my score: {gContext['result']}")
            break