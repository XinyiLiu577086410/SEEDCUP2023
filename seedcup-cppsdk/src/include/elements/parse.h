#pragma once
#include "json.hpp"
#include <cstdio>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using json = nlohmann::json;

namespace seedcup {

enum PacketType {
  INVALUE_PACKET = 0,
  INITREQ,
  ACTIONREQ,
  ACTIONRESP,
  GAMEOVERRESP,
};

enum ActionType {
  SILENT = 0, // 静止不动
  MOVE_LEFT,
  MOVE_RIGHT,
  MOVE_UP,
  MOVE_DOWN,
  PLACED, // 放置炸弹或者道具 相当于空格
};

enum ObjType {
  NULLOBJ = 0,
  PLAYER = 1,
  BOMB,
  BLOCK,
  ITEM,
};

enum ItemType {
  NULLITEM = 0,
  BOMB_RANGE,
  BOMB_NUM,
  HP,
  INVENCIBLE,
  SHIELD,
};

struct Player {
  int x;
  int y;
  bool alive;
  int bomb_max_num;
  int bomb_now_num;
  int bomb_range;
  int speed;
  int hp;
  int invincible_time;
  int player_id;
  std::string player_name;
  int score;
  int shield_time;
};

struct Bomb {
  int x;
  int y;
  int bomb_id;
  int bomb_range;
  int player_id;
};

struct Block {
  int x;
  int y;
  int block_id;
  bool removable;
};

struct Area {
  int x = -1;
  int y = -1;
  int bomb_id = -1;
  int block_id = -1;
  std::unordered_set<int> player_ids;
  ItemType item = NULLITEM;
};

struct GameMsg {
  int player_id;
  int game_round;
  /*           ~~~~key:id, value:item~~~~~~  */
  std::unordered_map<int, std::shared_ptr<Player>> players;
  std::unordered_map<int, std::shared_ptr<Bomb>> bombs;
  std::unordered_map<int, std::shared_ptr<Block>> blocks;
  std::vector<std::vector<Area>> grid;
};

inline std::pair<bool, std::shared_ptr<GameMsg>>
ParseMap(const std::string &input, int kMapSize) {
  json jsonData;
  try {
    jsonData = json::parse(input);
  } catch (const json::exception &e) {
    std::cerr << std::string(e.what()) << std::endl;
    return {false, nullptr};
  }

  std::shared_ptr<GameMsg> gameMsg = std::make_shared<GameMsg>();
  gameMsg->grid.resize(kMapSize, std::vector<Area>(kMapSize));

  gameMsg->player_id = jsonData["data"]["player_id"];
  gameMsg->game_round = jsonData["data"]["round"];

  for (const auto &item : jsonData["data"]["map"]) {
    Area area;
    area.x = item["x"];
    area.y = item["y"];

    if (item["objs"].is_array()) {
      for (const auto &obj : item["objs"]) {
        int type = obj["type"];

        if (type == PLAYER) {
          std::shared_ptr<Player> player = std::make_shared<Player>();
          player->alive = obj["property"]["alive"];
          player->bomb_max_num = obj["property"]["bomb_max_num"];
          player->bomb_now_num = obj["property"]["bomb_now_num"];
          player->bomb_range = obj["property"]["bomb_range"];
          player->hp = obj["property"]["hp"];
          player->invincible_time = obj["property"]["invincible_time"];
          player->player_id = obj["property"]["player_id"];
          player->score = obj["property"]["score"];
          player->shield_time = obj["property"]["shield_time"];
          player->speed = obj["property"]["speed"];
          player->player_name = obj["property"]["player_name"];
          player->x = area.x;
          player->y = area.y;

          // Map player ID to player object
          // You can store this in a global unordered map for easy access
          // For example: std::unordered_map<int, std::shared_ptr<Player>>
          // players; players[player->player_id] = player;
          gameMsg->players[player->player_id] = player;

          area.player_ids.insert(player->player_id);
        } else if (type == BOMB) {
          std::shared_ptr<Bomb> bomb = std::make_shared<Bomb>();
          bomb->bomb_id = obj["property"]["bomb_id"];
          bomb->bomb_range = obj["property"]["bomb_range"];
          bomb->player_id = obj["property"]["player_id"];
          bomb->x = area.x;
          bomb->y = area.y;

          // Map bomb ID to bomb object
          // You can store this in a global unordered map for easy access
          // For example: std::unordered_map<int, std::shared_ptr<Bomb>> bombs;
          // bombs[bomb->bomb_id] = bomb;
          gameMsg->bombs[bomb->bomb_id] = bomb;

          area.bomb_id = bomb->bomb_id;
        } else if (type == BLOCK) {
          std::shared_ptr<Block> block = std::make_shared<Block>();
          block->block_id = obj["property"]["block_id"];
          block->removable = obj["property"]["removable"];
          block->x = area.x;
          block->y = area.y;

          // Map block ID to block object
          // You can store this in a global unordered map for easy access
          // For example: std::unordered_map<int, std::shared_ptr<Block>>
          // blocks; blocks[block->block_id] = block;
          gameMsg->blocks[block->block_id] = block;

          area.block_id = block->block_id;
        } else if (type == ITEM) {
          area.item = obj["property"]["item_type"];
        }
      }
    }

    gameMsg->grid[area.x][area.y] = area;
  }

  return {true, gameMsg};
}

inline bool ParseWinner(const std::string &json_str,
                        std::vector<std::pair<int, int>> &scores,
                        std::vector<int> &winners) {
  json jsonData;
  try {
    jsonData = json::parse(json_str);
  } catch (const json::exception &e) {
    return false;
  }
  for (const auto &item : jsonData["data"]["scores"]) {
    scores.push_back({item["player_id"], item["score"]});
  }
  for (const auto &item : jsonData["data"]["winner_ids"]) {
    winners.push_back(item);
  }
  return true;
}

inline PacketType ParsePacket(const std::string &json_str) {
  json json_data;
  try {
    json_data = json::parse(json_str);
  } catch (const json::exception &e) {
    return INVALUE_PACKET;
  }
  return PacketType(json_data["type"]);
}

} // namespace seedcup
