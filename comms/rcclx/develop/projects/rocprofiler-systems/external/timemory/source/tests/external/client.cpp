
#include "timemory/utility/socket.hpp"

#include <chrono>
#include <string>
#include <thread>
#include <vector>

int
main(int argc, char** argv)
{
    auto socket_manager = tim::socket::manager{};
    socket_manager.connect("send", "127.0.0.1", 8080);
    if(argc == 1)
    {
        std::string input;
        while(true)
        {
            std::cout << std::flush << "> " << std::flush;
            std::getline(std::cin, input);
            if(input.empty())
                break;
            if(!socket_manager.send("send", input))
                break;
        }
    }
    else
    {
        for(int i = 1; i < argc; ++i)
        {
            if(!socket_manager.send("send", argv[i]))
                break;
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    socket_manager.close("send");

    return 0;
}
