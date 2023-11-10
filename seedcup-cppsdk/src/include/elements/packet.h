#pragma once
#include "parse.h"

namespace seedcup {

inline std::string CreateInitPacket(const std::string &player_name) {
  json json_obj;
  json_obj["type"] = int(INITREQ);
  json_obj["data"] = {
      {"player_name", player_name},
  };
  return json_obj.dump();
}

inline std::string CreateActionPacket(int playerID, ActionType action) {
  json json_obj;
  json_obj["type"] = int(ACTIONREQ);
  json_obj["data"] = {
      {"playerID", playerID},
      {"actionType", (int)action},
  };
  return json_obj.dump();
}

} // namespace seedcup
