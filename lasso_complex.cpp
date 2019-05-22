#include <math.h>
#include "lasso_complex.h"


complex operator+(complex a, complex b){
    complex c;
    c.real = a.real+b.real;
    c.imag = a.imag+b.imag;
    return c;
}

complex operator-(complex a, complex b){
    complex c;
    c.real = a.real-b.real;
    c.imag = a.imag-b.imag;
    return c;
}

complex operator*(complex a, complex b){
    complex c;
    float r, i;
    r = a.real*b.real-a.imag*b.imag;
    i = a.imag*b.real+a.real*b.imag;
    c.real = r;
    c.imag = i;
    return c;
}

complex operator/(complex a, complex b){
    complex c;
    float r, i, d;
    r = a.real*b.real+a.imag*b.imag;
    i = a.real*b.imag-a.imag*b.real;
    d = a.real*a.real+b.real*b.real;
    c.real = r/d;
    c.imag = i/d;
    return c;
}


complex cexp(complex a){
    complex c;
    float t = exp(a.real);
    c.real = t*cos(a.imag);
    c.imag = t*sin(a.imag);
    return c;
}

float cabs(complex a){
    return sqrt(a.real*a.real+a.imag*a.imag);
}

complex cpow(complex a, float b){
    complex c;
    float s = cabs(a);
    float t = pow(s, b);
    float u = acos(a.real/s)*b;

    c.real = t*cos(u);
    c.imag = t*sin(u);
    return c;
}