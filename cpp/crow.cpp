/*
 * @Author: yangyahe yangyahe@midu.com
 * @Date: 2024-10-13 15:46:06
 * @LastEditors: yangyahe yangyahe@midu.com
 * @LastEditTime: 2024-10-13 15:52:01
 * @FilePath: /app/yangyahe/yyh-util/cpp/crow.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <crow.h>

int main(){
    crow::SimpleApp app;
    //测试
     CROW_ROUTE(app, "/test")([](){
        return "Hello world";
    });
    app.port(18888).multithreaded().run();
}

// g++ main.cpp -o  main -lpthread 
