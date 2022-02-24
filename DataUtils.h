#include <CkTar.h>
#include <iostream>

#pragma once
class DataUtils
{

public:static    void Untar(std::string path)
    {
        CkTar tar;
        tar.put_UntarFromDir("data/input");
        tar.put_NoAbsolutePaths(true);
        bool success = tar.UntarBz2(path.c_str());
        if (success != true) {
            std::cout << tar.lastErrorText() << "\r\n";
        }
        else {
            std::cout << "Success" << "\r\n";
        }
    }
};

