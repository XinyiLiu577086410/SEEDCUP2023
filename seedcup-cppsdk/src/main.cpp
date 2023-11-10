#include "include/elements/parse.h"
#include "include/seedcup.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
using namespace seedcup;
#define DEMO_FLAG
#define Nmap 15
#define MY_NAME "gttaVX/u5z5AmTBRenaVDeEmUyrCUKA//YZtLOiaF/k="
static bool Reachable[Nmap][Nmap];
static int Distance[Nmap][Nmap];
Area Map[Nmap][Nmap];

static void dfs(GameMsg & msg, int x, int y) {
  if(msg.blocks[Map[x][y].block_id]->removable);
}

void TakeMyAction(GameMsg & msg, vector<ActionType>& MyAction) {
  do{
    int l1 = msg.grid.size(), l2 = msg.grid[0].size();
    for(int i = 0; i < l1; i++) {
      for(int j = 0; j < l2; j++) {
        Map[i][j] = msg.grid[i][j];
      }
    }
  } while(0);
  for(auto & p : msg.players) {
    if(p.second->player_name == MY_NAME)
      dfs(msg, p.second->x,p.second->y);
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
  seedcup.RegisterCallBack(

      [](GameMsg &msg, SeedCup &server) -> int {
        
        #ifdef DEMO_FLAG
        {
        // 打印自己的id
        std::cout << "self:" << msg.player_id << std::endl;

        // 打印地图
        cout << endl << endl;
        auto &grid = msg.grid;
        for (int i = 0; i < msg.grid.size(); i++) {
          for (int j = 0; j < msg.grid.size(); j++) {
            auto &area = grid[i][j];
            if (grid[i][j].block_id != -1) {
              if (msg.blocks[grid[i][j].block_id]->removable) {
                cout << "0 ";
              } else {
                cout << "9 ";
              }
            } else if (area.player_ids.size() != 0) {
              cout << "1 ";
            } else if (area.bomb_id != -1) {
              cout << "8 ";
            } else if (area.item != seedcup::NULLITEM) {
              switch (area.item) {
              case seedcup::BOMB_NUM:
                cout << "a ";
                break;
              case seedcup::BOMB_RANGE:
                cout << "b ";
                break;
              case seedcup::INVENCIBLE:
                cout << "c ";
                break;
              case seedcup::SHIELD:
                cout << "d ";
                break;
              case seedcup::HP:
                cout << "e ";
                break;
              default:
                cout << "**";
                break;
              }
            } else {
              cout << "__";
            }
          }
          cout << endl;
        }
        return server.TakeAction((ActionType)(5));
        }
        // #endif
        #else
        vector<ActionType> MyAction;
        TakeMyAction(msg, MyAction);
        for(auto p : MyAction) {
          int status;
          if(status = server.TakeAction(p))
            return status;
        }
        return 0;
        #endif
      },
      [](int player_id, const std::vector<std::pair<int, int>> &scores, const std::vector<int> &winners) -> int {
        // 打印所有人的分数
        for (int i = 0; i < scores.size(); i++) {
          cout << "[" << scores[i].first << "," << scores[i].second << "] ";
        }
        // 打印获胜者列表
        cout << endl;
        for (int i = 0; i < winners.size(); i++) {
          cout << winners[i] << " ";
        }
        cout << endl;
        return 0;
      });
  ret = seedcup.Run();
  if (ret != 0) {
    std::cout << seedcup.get_last_error();
    return ret;
  }
  return 0;
}
