#include "src.h"
#include <iostream>
#define print(x) {std::cout << x << std::endl;}



int main(){
    // 
    int output = fact(5);
    print(output);
    
    for (int i = 0; i != 10; ++i){
        std::cout << static_variable() << std::endl;
    }
    extern int x;
    print(x);

}
