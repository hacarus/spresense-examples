#include <math.h>

struct complex_
{
    float real;
    float imag;
};
typedef struct complex_ complex;


const complex J = {0, 1};

complex operator+(const complex_ a, const complex_ b);
complex operator-(const complex_ a, const complex_ b);
complex operator*(const complex_ a, const complex_ b);
complex operator/(const complex_ a, const complex_ b);
complex cexp(complex a);
float cabs(complex a);
complex cpow(complex a, float b);