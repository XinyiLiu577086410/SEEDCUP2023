#pragma once
#include <cstring>
#include <iostream>
#include <ostream>
#include <string>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/epoll.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#else
#include <winsock2.h>
#endif

class SocketApi {
private:
  SocketApi() = delete;
  SocketApi(const SocketApi &) = delete;

public:
  static inline int setSocWriteWait(int fd, unsigned sec, unsigned msec = 0) {
    if (sec == 0 && msec == 0)
      return 0;
    struct timeval tv;
    tv.tv_sec = sec;
    tv.tv_usec = msec;
#ifndef _WIN32
    return setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(struct timeval));
#else
    return setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char *)&tv,
                      sizeof(struct timeval));
#endif
  }
  static inline int setSockWriteSize(int fd, int size) {
#ifndef _WIN32
    return setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &size, sizeof(int));
#else
    return setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (const char *)&size,
                      sizeof(int));
#endif
  }
  static inline int setSockReadSize(int fd, int size) {
#ifndef _WIN32
    return setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &size, sizeof(int));
#else
    return setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (const char *)&size,
                      sizeof(int));
#endif
  }
  static inline int setSocReadWait(int fd, unsigned sec, unsigned msec = 0) {
    if (sec == 0 && msec == 0)
      return 0;
    struct timeval tv;
    tv.tv_sec = sec;
    tv.tv_usec = msec;
#ifndef _WIN32
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(struct timeval));
#else
    return setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv,
                      sizeof(struct timeval));
#endif
  }
  static inline int setDeferAccept(int fd, int timeout = 5) {
#ifndef _WIN32
    return setsockopt(fd, IPPROTO_TCP, TCP_DEFER_ACCEPT, &timeout, sizeof(int));
#else
    return setsockopt(fd, IPPROTO_TCP, TCP_DEFER_ACCEPT, (const char *)&timeout,
                      sizeof(int));
#endif
  }
  static inline int receiveSocket(int fd, void *buffer, size_t len,
                                  int flag = 0) {
    return recv(fd, (char *)buffer, len, flag);
  }
  static int recvSockBorder(int fd, std::string &buffer, const char *border,
                            int flag = 0) {
    if (border == NULL || strlen(border) == 0)
      return -1;
    char firstCh = border[0], now = 0, temp[1024] = {0};
    char *left = (char *)malloc(sizeof(char) * strlen(border));
    if (left == NULL)
      return -1;
    memset(left, 0, sizeof(char) * strlen(border));
    int len = 0, all = 0, leftsize = strlen(border) - 1;
    while (1) {
      len = recv(fd, (char *)&now, 1, flag);
      temp[all] = now;
      all += 1;
      if (all >= 1024) {
        buffer += std::string(temp, 1024);
        all = 0;
      }
      if (len <= 0)
        return -1;
      if (now == firstCh) {
        len = recv(fd, (char *)left, leftsize, MSG_PEEK);
        if (len <= 0)
          return -1;
        if (strncmp(left, border + 1, leftsize) == 0) {
          len = recv(fd, (char *)left, leftsize, flag);
          buffer += std::string(temp, all);
          buffer += now;
          buffer += left;
          break;
        }
      }
    }
    free(left);
    return buffer.size();
  }
  static int recvSockSize(int fd, std::string &buffer, size_t needSize,
                          int flag = 0) {
    if (needSize == 0)
      return -1;
    char *pbuffer = (char *)malloc(sizeof(char) * needSize + 1);
    if (pbuffer == NULL)
      return -1;
    memset(pbuffer, 0, sizeof(char) * needSize + 1);
    int len = recv(fd, pbuffer, needSize, MSG_WAITALL | flag);
    if (len <= 0)
      return len;
    buffer += std::string(pbuffer, len);
    free(pbuffer);
    return len;
  }
  static int receiveSocket(int fd, std::string &buffer, int flag = 0) {
    char temp[1024] = {0};
    int len = 0;
    do {
      memset(temp, 0, sizeof(char) * 1024);
      len = recv(fd, (char *)temp, 1024, flag);
      if (len > 0)
        buffer += std::string(temp, len);
      else
        break;
    } while (len == 1024);
    return buffer.size() > 0 ? buffer.size() : len;
  }
  static inline int sendSocket(int fd, const void *buffer, size_t len,
                               int flag = 0) {
    return send(fd, (char *)buffer, len, flag);
  }
  static inline int tryConnect(int fd, const sockaddr *addr, socklen_t len) {
    auto ret = connect(fd, addr, len);
    return ret;
  }
  static inline int bindSocket(int fd, const sockaddr *addr, socklen_t len) {
    return bind(fd, addr, len);
  }
  static inline int acceptSocket(int fd, sockaddr *addr, socklen_t *plen) {
    return accept(fd, addr, plen);
  }
  static inline int listenSocket(int fd, int backLog) {
    return listen(fd, backLog);
  }
  static inline int closeSocket(int fd) {
#ifndef _WIN32
    return close(fd);
#else
    return closesocket(fd);
#endif
  }
};

class ClientTcpIp {
private:
  int sock;          // myself
  sockaddr_in addrC; // server information
  std::string connectIP;
  std::string selfIP;
  const char *error;
  bool isHttps;

public:
  ClientTcpIp() {
    error = NULL;
    isHttps = false;
    sock = socket(AF_INET, SOCK_STREAM, 0);
    addrC.sin_family = AF_INET; // af_inet IPv4
  }
  ClientTcpIp(const std::string &connectIP, unsigned short connectPort)
      : ClientTcpIp() {
    this->connectIP = connectIP;
    addrC.sin_addr.s_addr = inet_addr(connectIP.c_str());
    addrC.sin_port = htons(connectPort);
  }
  ~ClientTcpIp() {
#ifndef _WIN32
    close(sock);
#else
    closesocket(sock);
#endif
  }
  void addHostIp(const std::string &connectIP, unsigned short port = 0) {
    this->connectIP = connectIP;
    addrC.sin_addr.s_addr = inet_addr(connectIP.c_str());
    if (port != 0)
      addrC.sin_port = htons(port);
  }
  bool tryConnect() {
    if (SocketApi::tryConnect(sock, (sockaddr *)&addrC, sizeof(sockaddr)) ==
        -1) {
      error = "connect wrong";
      return false;
    }
    return true;
  }
  inline int getSocket() { return sock; }
  inline int receiveHost(void *prec, int len, int flag = 0) {
    return SocketApi::receiveSocket(sock, prec, len, flag);
  }
  int receiveHost(std::string &buffer, int flag = 0) {
    return SocketApi::receiveSocket(sock, buffer, flag);
  }
  inline int sendHost(const void *ps, int len, int flag = 0) {
    return SocketApi::sendSocket(sock, ps, len, flag);
  }
  inline int sendHost(const std::string &data, int flag = 0) {
    return SocketApi::sendSocket(sock, data.data(), data.size(), flag);
  }
  bool disconnectHost() {
#ifndef _WIN32
    close(sock);
#else
    closesocket(sock);
#endif
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1)
      return false;
    return true;
  }
  std::string getSelfIp() {
    char name[300] = {0};
    gethostname(name, 300);
    hostent *phost = gethostbyname(name);
    in_addr addr;
    char *p = phost->h_addr_list[0];
    memcpy(&addr.s_addr, p, phost->h_length);
    return inet_ntoa(addr);
  }
  std::string getSelfName() {
    char name[300] = {0};
    int result = gethostname(name, 300);
    if (result != 0) {
      error = "get host name wrong";
      return "";
    }
    return name;
  }
  inline const char *lastError() { return error; }

public:
  static std::string getDnsIp(const std::string &domain) {
    hostent *phost = gethostbyname(domain.c_str());
    if (phost == nullptr) {
      return "";
    }
    in_addr addr;
    char *p = phost->h_addr_list[0];
    memcpy(&addr.s_addr, p, phost->h_length);
    return inet_ntoa(addr);
  }
};
