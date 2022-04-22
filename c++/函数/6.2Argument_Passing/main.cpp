#include "src.h"
#include <iostream>
#define print(x) {std::cout << x << std::endl;}



int main(){
    // passing argument by value
    int value = 100;
    // call the function
    change_argument(value);
    // print the argument
    print(value);


    // Pointer Parameters



    int n = 0;
    // int i = n; // i是n的一个拷贝
    // i = 2; // 给i重新赋值
    print(n); // i的结果是2


    print("Hello")
}
