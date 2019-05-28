// #include <complex.h>
#include <math.h>
#include <stdlib.h>
#include "lasso_ndarray.h"
#include "lasso_fft.h"
// #include "lasso_complex.h"

const float PI = 3.14159265358979;
uint bit_reverse(uint data, int &i){
    data = ((data & 0x55555555) << 1)|((data & 0xAAAAAAAA) >> 1);
    data = ((data & 0x33333333) << 2)|((data & 0xCCCCCCCC) >> 2);
    data = ((data & 0x0F0F0F0F) << 4)|((data & 0xF0F0F0F0) >> 4);
    data = ((data & 0x00FF00FF) << 8)|((data & 0xFF00FF00) >> 8);
    data = (data << 16)|(data >> 16);
    return data>>32-i;
}

uint bit_count(uint data){
    data = (data & 0x55555555)+((data & 0xAAAAAAAA) >> 1);
    data = (data & 0x33333333)+((data & 0xCCCCCCCC) >> 2);
    data = (data & 0x0F0F0F0F)+((data & 0xF0F0F0F0) >> 4);
    data = (data & 0x00FF00FF)+((data & 0xFF00FF00) >> 8);
    data = (data & 0X0000FFFF)+((data & 0xFFFF0000) >> 16);
    return data;
}

template<typename T>
inline void swap(T &a, T &b){
    T temp = a;
    a = b;
    b = temp;
}

inline void butterfly(complex &a, complex &b, complex w){
    complex bw = b * w;
    b = a - bw;
    a = a + bw;
}

complex *fft(complex *a, int N)
{
    complex *freq = (complex *)calloc(N, sizeof(complex));
    int bit = bit_count(N-1);
    complex t = {0, -2*PI / N};
    complex W = cexp(t);
    
    int j_ = 1 << bit-1;
    int k_ = 1;

    for (int i = 0; i < N; ++i)
    { 
        freq[i] = a[bit_reverse(i, bit)];
    }

    for(int i = 0; i < bit; ++i){
        for(int j = 0; j < j_; ++j){
            for (int k = 0; k < k_; ++k)
            {
                butterfly(freq[j*k_*2+k], freq[j*k_*2+k+k_], cpow(W, j_*k));
            }
        }
        k_ <<= 1;
        j_ >>= 1;
    }
    return freq;
}


complex* ifft(complex *a, int N){
    complex *freq = (complex*)calloc(N, sizeof(complex));
    
    int bit = bit_count(N-1);
    complex t = {0, -2*PI / N};
    complex W = cexp(t);
    
    int j_ = 1 << bit-1;
    int k_ = 1;

    for (int i = 0; i < N; ++i)
    {
        freq[i] = a[bit_reverse(i, bit)];
    }

    for(int i = 0; i < bit; ++i){
        for(int j = 0; j < j_; ++j){
            for (int k = 0; k < k_; ++k)
            {
                butterfly(freq[j*k_*2+k], freq[j*k_*2+k+k_], cpow(W, -j_*k));
            }
        }
        k_ <<= 1;
        j_ >>= 1;
    }
    for (int i = 0; i < N; ++i)
    {
        freq[i].real /= N;
        freq[i].imag /= N;
    }
    
    return freq;
}
