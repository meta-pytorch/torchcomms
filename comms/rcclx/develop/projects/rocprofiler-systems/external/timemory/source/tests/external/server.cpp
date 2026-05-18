
#include "timemory/utility/socket.hpp"

#include <string>
#include <vector>

int
main()
{
    std::vector<std::string> results;
    auto                     handle_data = [&](std::string str) {
        std::cout << "client sent: " << str << std::endl;
        results.emplace_back(std::move(str));
    };
    std::cout << "server is listening..." << std::endl;
    tim::socket::manager{}.listen("test", 8080, handle_data);
    std::cout << "server stopped listening..." << std::endl;

    std::cout << "server data:\n";
    for(auto& itr : results)
        std::cout << "    " << itr << '\n';
    return 0;
}
