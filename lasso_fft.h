#include "lasso_complex.h"

uint bit_count(uint data);
uint bit_reverse(uint data, int &i);
complex *fft(complex *a, int N);
complex* ifft(complex *a, int N);