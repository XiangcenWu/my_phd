#include "src.h"


int fact(int val){
    int ret = 1;
    while (val > 1){
        ret *= val--; // 将ret*val赋值给ret后再减一
    }
    return ret;
}

int static_variable(){
    static int sv = 0;
    return ++sv;
}

int x = 1232;