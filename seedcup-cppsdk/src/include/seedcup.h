#pragma once
#include "config.h"
#include "elements/packet.h"
#include "elements/parse.h"
#include "net/client.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace seedcup {

class SeedCup;
using MapCallBack = std::function< int (GameMsg &, SeedCup &) >;
using GameOverCallBack = std::function<  int  ( int player_id, const std::vector<std::pair<int, int>> &scores, const std::vector<int> &winners)  >;

class SeedCup final {
public:
  SeedCup(const std::string &config_path, const std::string &player_name) : client_(), player_name_(player_name) 
  {
    Config::set_path(config_path);
  }
  ~SeedCup()  
  { 
    client_.disconnectHost(); 
  }
  int Init() 
  {
    std::string host = Config::get_instance().get<std::string>("host");
    int port = Config::get_instance().get<unsigned short>("port");
    client_.addHostIp(host, port);
    if (!client_.tryConnect()) 
    {
      last_error_ = "connect host " + host + 
                    ":" + std::to_string(port) +
                    " " + client_.lastError() + 
                    " " + std::to_string(errno);
      return -1;
    }
    return 0;
  }
  int Run() 
  {
    auto init_packet = CreateInitPacket(player_name_);
    if (sendData(init_packet) < 0) return -1;
    std::string data;
    const int kMapSize = Config::get_instance().get<int>("map_size");
    bool isContinue = true;
    std::vector<std::pair<int, int>> scores;
    std::vector<int> winners;
    while (isContinue && recvData(data) >= 0)
    {
      auto packet_type = ParsePacket(data);
      std::pair<bool, std::shared_ptr<GameMsg>> action_result;
      switch (packet_type) 
      {
      case ACTIONRESP:
        action_result = ParseMap(data, kMapSize);
        if (!action_result.first) 
        {
          last_error_ = "parse map wrong";
          isContinue = false;
        } 
        else 
        {
          this->player_id_ = action_result.second->player_id;
          if (map_call_back_ != nullptr) 
          {
            auto ret = map_call_back_(*action_result.second, *this);
            if (ret != 0) 
            {
              last_error_ = "ret not zero " + std::to_string(ret);
              isContinue = false;
            }
          }
        }
        break;
      case GAMEOVERRESP:
        if (!ParseWinner(data, scores, winners)) 
        {
          last_error_ = "parse game over wrong";
        } 
        else 
        {
          if (over_call_back_ != nullptr) 
          {
            auto ret = over_call_back_(this->player_id_, scores, winners);
            if (ret != 0) 
            {
              last_error_ = "ret not zero " + std::to_string(ret);
            }
          }
        }
        isContinue = false;
        break;
      default:
        last_error_ = "get wrong packet type";
        break;
      }
    }
    client_.disconnectHost();
    return 0;
  }

  int TakeAction(ActionType action_type) 
  {
    auto result = CreateActionPacket(player_id_, action_type);
    return sendData(result);
  }

  inline const std::string &get_last_error() { 
    return last_error_; 
  }

  void RegisterCallBack(MapCallBack map_call, GameOverCallBack game_over_call) 
  {
    map_call_back_ = map_call;
    over_call_back_ = game_over_call;
  }

  template <typename Valuetype> Valuetype GetConfig(const std::string &key) 
  {
    return Config::get_instance().get<Valuetype>(key);
  }

private:
  int sendData(const std::string &data)
  {
    uint64_t size = data.size();
    char buff[sizeof(size) + data.size()];
    memcpy(buff, &size, sizeof(size));
    memcpy(buff + sizeof(size), data.data(), data.size());
    auto ret = client_.sendHost(buff, sizeof(buff));
    if (ret < 0) 
    {
      this->last_error_ = "send host body wrong " + std::to_string(ret) +
                          " errno:" + std::to_string(errno);
      return ret;
    }
    return 0;
  }

  int recvData(std::string &data) 
  {
    uint64_t size = 0;
    auto ret = client_.receiveHost(&size, 8);
    if (ret <= 0) 
    {
      this->last_error_ = "recv host len wrong " + std::to_string(ret);
      return -1;
    }

    if (size > 65536) 
    {
      this->last_error_ = "recv host len to large " + std::to_string(size);
      return -1;
    }

    char *buffer = new char[size];

    int bytesReceived = 0;
    while (bytesReceived < size) 
    {
      // 调用 recv() 函数接收数据
      int ret =
          client_.receiveHost(buffer + bytesReceived, size - bytesReceived);
      if (ret <= 0) 
      {
        // 接收失败或连接断开
        this->last_error_ = "send host body wrong " + std::to_string(ret);
        return -1;
      }
      bytesReceived += ret;
    }

    // 将接收到的数据存储到 std::string 对象中
    data.assign(buffer, size);

    delete[] buffer;

    return 0; // 返回成功
  }

private:
  MapCallBack map_call_back_;
  GameOverCallBack over_call_back_;
  std::string last_error_;
  std::string player_name_;
  ClientTcpIp client_;
  int player_id_;
};

} // namespace seedcup
