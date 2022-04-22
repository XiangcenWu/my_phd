#include "src.h"
#include <iostream>
#define print(x) {std::cout << x << std::endl;}

void add(int x){
        x += 1;
    }

int main(){
    int s = 10;
    add(s);
    print(s);
    
}
