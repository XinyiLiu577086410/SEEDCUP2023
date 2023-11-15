import json
import socket
from base import *
from req import *
from resp import *
from config import config
from ui import UI
import subprocess
from logger import logger
import sys
	
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

#from cdk
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


# user function
def getState(resp : PacketResp):
	for grid in resp.data.map:
		if len(grid.objs):
			for obj in grid.objs:
				if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
					state = [obj.property.hp, obj.property.bomb_range, obj.property.shield_time, obj.property.invincible_time]
	for grid in resp.data.map:
		if len(grid.objs):
			state.append(grid.objs[0].type)
		else:
			state.append(0)
	return state
	# raise Exception("Invalid State")


# user function
def calcReward(resp1 : PacketResp, resp2 : PacketResp):
	score1 = 0
	score2 = 0
	for grid in resp1.data.map:
		if len(grid.objs):
			for obj in grid.objs:
				if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
					score1 = obj.property.score
	if resp2.type != PacketType.GameOver:
		for grid in resp2.data.map:
			if len(grid.objs):
				for obj in grid.objs:
					if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
						score2 = obj.property.score
	else:
		for res in resp2.data.scores:
			if res["player_id"] == gContext["playerID"]:
				score2 = res["score"]
				gContext["result"] = res["score"]

	return score2-score1
# user function
def termPlayAPI():
	"""
	Connects to the client, sends an initialization packet, and receives a response packet.
	
	Returns:
		client (Client): The connected client object.
		stat_resp (PacketResp): The response packet received from the client.
	"""
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
	for grid in resp.data.map:
		if grid.x == pos[0] and grid.y == pos[1]:
			if len(grid.objs):
				for obj in grid.objs:
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
	if resp.type != PacketType.GameOver:			
		for grid in resp.data.map:
			if len(grid.objs):
				for obj in grid.objs:
					if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
						# print(f"Start Search at {grid.x},{grid.y}\n")
						return search([grid.x, grid.y], [grid.x, grid.y], resp, type)
	return False

# user function
# not completed? 
def canMove(resp : PacketResp):
	return True

# user function
# 
def bombPutted(resp : PacketResp):
	if resp.type == PacketType.ActionResp:
		for grid in resp.data.map:
			if len(grid.objs):
				for obj in grid.objs:
					if obj.type == ObjType.Player:
						if obj.property.player_id == gContext["playerID"]:
							if obj.property.bomb_now_num != 0:
								return True
							else:
								return False
		raise Exception("Not Found!")


# user function
def inArea(resp : PacketResp):
	for grid in resp.data.map:
		if len(grid.objs):
			for obj in grid.objs:
				if obj.type == ObjType.Bomb:
					for i in range(obj.property.bomb_range):
						for direction in range(5):
							xx = grid.x + direct[direction][0] * (i+1)
							yy = grid.y + direct[direction][1] * (i+1)
							if xx >= 0 and xx < 15 and yy >=0 and yy < 15:
								for grid2 in resp.data.map:
									if grid2.x == xx and grid2.y == yy and len(grid2.objs):
										for obj2 in grid2.objs:
											if obj2.type == ObjType.Player and obj2.property.player_id == gContext["playerID"]:
												return [grid.x, grid.y, direction, i]
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
	for grid in resp.data.map:
		if len(grid.objs):
			for obj in grid.objs:
				if obj.type == ObjType.Player and obj.property.player_id == gContext["playerID"]:
					pos.extend([grid.x, grid.y])
	# match(direct):
	if direct == 0:
		for grid in resp.data.map:
			if grid.x == pos[0] and grid.y == pos[1] -1: #left block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] and grid.y == pos[1] +1: #right block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] -1 and grid.y == pos[1]: #up block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
	elif direct == 1:
		for grid in resp.data.map:
			if grid.x == pos[0] -1 and grid.y == pos[1]: #up block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] +1 and grid.y == pos[1]: #down block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] and grid.y == pos[1] +1: #right block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
	elif direct == 2:
		for grid in resp.data.map:
			if grid.x == pos[0] and grid.y == pos[1] -1: #left block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] and grid.y == pos[1] +1: #right block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] +1 and grid.y == pos[1]: #down block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
	elif direct == 3:
		for grid in resp.data.map:
			if grid.x == pos[0] -1 and grid.y == pos[1]: #up block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] +1 and grid.y == pos[1]: #down block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] and grid.y == pos[1] -1: #left block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
	elif direct == 4:
		for grid in resp.data.map:
			if grid.x == pos[0] and grid.y == pos[1] -1: #left block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] and grid.y == pos[1] +1: #right block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] -1 and grid.y == pos[1]: #up block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
			if grid.x == pos[0] +1 and grid.y == pos[1]: #down block
				if not len(grid.objs):
					return [[grid.x,grid.y], pos]
	return []

# user function
def Stage1Play(client : Client, resp : PacketResp):
	

def Stage2Play(client : Client, resp : PacketResp):
	pass

if __name__ == "__main__":
	client, resp1 = termPlayAPI()
	Stage1Play(client)
	Stage2Play(client)