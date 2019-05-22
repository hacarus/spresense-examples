#include "lasso_xorshift.h"

const long long LLONG_MAX = 9223372036854775807;
long long seed0=364258948240557689764, seed1=2251614694823310722, seed2=808714612742409123456, seed3=300423169847275242345678;
long long xor128() {
    long long t=(seed0^(seed0<<11));
    seed0=seed1; seed1=seed2; seed2=seed3;
    return ( seed3=(seed3^(seed3>>19))^(t^(t>>8)) );
}

template<typename T>
T my_rand(T range){
    // return range * xor128() / LLONG_MAX;
    return xor128() / (LLONG_MAX / range);
}
template int my_rand<int>(int);
template float my_rand<float>(float);
template double my_rand<double>(double);
template long long my_rand<long long>(long long);

