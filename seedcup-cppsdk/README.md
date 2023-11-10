# 种子杯2023 cpp-sdk

## 创建人

- chenxuan

## 使用说明

- 需要安装cmake,g++,make

- 操作系统最好为linux(理论上win+MinGw也可以,但是没有测试在window上编译)

## 编译说明

1. `git clone https://gitee.com/chenxuan520/seedcup-cppsdk`

2. `cd seedcup-cppsdk/src`

3. `mkdir build;cd build;cmake ..;make`

4. 会在src目录下生成bin文件夹.二进制文件在里面

## 开发说明
> 核心参考main.cpp,这是一个简单的demo

1. 包含头文件`include/seedcup.h`,以及使用命名空间`using namespace seedcup`

2. 创建seedcup类(需要传入配置文件路径以及玩家名字)以及初始化`seedcup.Init()`

	- Init函数返回值为是否成功(0为成功),如果不为0,使用`seedcup.get_last_error()`函数得到错误信息

3. 调用RegisterCallBack函数,创建消息回调处理

	- 第一个函数参数为`std::function<GameMsg&, SeedCup &)>`,当服务器下发地图时候会触发

		- 第一个参数传入地图中的信息

		- 第二个参数传入本体seedcup,可以使用seedcup的TakeAction函数先服务器发送你的操作

	- 第二个函数参数为`std::function<int(int player_id, const std::vector<std::pair<int, int>> &scores, const std::vector<int> &winners)>`,游戏结束时候下发

	- 函数参数的返回值int标识错误,如果不为0,`seedcup.Run()`会返回

4. 调用`seedcup.Run()`,该函数不会返回,除非游戏结束或者发生内部错误(网络错误或者,回调函数发出错误)
