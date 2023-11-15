#include "include/elements/parse.h"
#include "include/seedcup.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>
using namespace std;
using namespace seedcup;
#define INF 0x7f7f7f7f
#define MY_NAME "gttaVX/u5z5AmTBRenaVDeEmUyrCUKA//YZtLOiaF/k="
struct route{
  int len = INF;
  vector<pair<int,int>> path;
  vector<pair<int,int>> next;
};


struct path_t{
  map<pair<int,int>, route> routes;
} analyzeRes;

// return shortest path
path_t Dijstra(GameMsg & msg) {
  path_t res;
  /* algorithm */
  return res;
}



/* Indentification Friend or Foe */
void IFF(GameMsg & msg, set<Player>& enemy, set<Player>& engagedEnemy, path_t& paths) {
  Player self = *msg.players[msg.player_id];
  for(auto p : msg.players) {
    if(p.first == msg.player_id) continue;
    else enemy.insert(*p.second);
  }
  for(auto p : enemy) {
    if(paths.routes[{p.x, p.y}].len != INF){
      engagedEnemy.insert(p);
    }
  }

}

void SelectBlock(GameMsg & msg, set<Player>& enemy, set<Player>& engagedEnemy, path_t& paths, Block& nearestBlock) {
  int nearestLen = INF;
  for(auto p : msg.blocks){
    Block thisBlock = *p.second;
    if(paths.routes[{thisBlock.x,thisBlock.y}].len < nearestLen) {
      nearestBlock = thisBlock;
      nearestLen = paths.routes[{thisBlock.x,thisBlock.y}].len;
    }
  }
}


void GoTo(){}

void TakeMyAction(GameMsg & msg, vector<ActionType>& MyAction) {
  
  path_t paths = Dijstra(msg);
  Player self = *msg.players[msg.player_id];
  set<Player> enemy, engagedEnemy;
  IFF(msg, enemy, engagedEnemy, paths);
  static Block nearestBlock;
  static bool onTheWay = false;
  static int lastStep = 0;
  if(engagedEnemy.size() == 0) {
    if(onTheWay == false) {
      SelectBlock(msg, enemy, engagedEnemy, paths, nearestBlock);
      lastStep == 0;
      onTheWay = true;
    }
  } 
  else {

  }
}

int main() {
  SeedCup seedcup("../config.json", MY_NAME);
  int ret = seedcup.Init();
  if (ret != 0) {
    std::cout << seedcup.get_last_error();
    return ret;
  }
  cout << "init client success" << endl;
  seedcup.RegisterCallBack
  (
    /* argument 1 */
    [](GameMsg &msg, SeedCup &server) -> int { 
      vector<ActionType> MyAction;
      TakeMyAction(msg, MyAction);
      for(auto p : MyAction) {
        int status;
        if(status = server.TakeAction(p))
          return status; /* report exception */
      }
      return 0;
    }
    ,
    /* argument 2 */
    [](int player_id, const std::vector<std::pair<int, int>> &scores, const std::vector<int> &winners) -> int {
        /* 打印所有人的分数 */
        for (int i = 0; i < scores.size(); i++) {cout << "[" << scores[i].first << "," << scores[i].second << "] ";}
        /* 打印获胜者列表 */
        cout << endl; for (int i = 0; i < winners.size(); i++) {cout << winners[i] << " ";}
        cout << endl;return 0;
      }
  );
  ret = seedcup.Run();
  if (ret != 0) {
    std::cout << seedcup.get_last_error();
    return ret;
  }
  return 0;
}
