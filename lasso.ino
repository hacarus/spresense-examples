#include "lasso_ndarray.h"
#include "lasso_xorshift.h"
#include "lasso_train.h"
#include "lasso_fft.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

char my_getchar(){
    while(!Serial.available()){
        delay(1);
    }
    return Serial.read();
}

int read_int(){
    int n = 0;
    while(true){
        char t = my_getchar();
        if(48 <= t && t <= 57){
            n *= 10;
            n += t-48;
        }else{
            return n;
        }
    }
}

float read_num(){
    float n = 0;
    float o = 1;
    int d = false;
    while(true){
        char t = my_getchar();

        if(48 <= t && t <= 57){
            n *= 10;
            n += t-48;
            if(d){
                o *= 10;
            }
        }else if(t == '-'){
            o*=-1;
        }else if(t == '.'){
            d = true;
        }else{
            return n/o;
        }
    }
}

ndarray load_img(){
    int size[2];
    Serial.print("x size:");
    size[0] = read_int();
    Serial.println(size[0]);
    Serial.print("y size:");
    size[1] = read_int();
    Serial.println(size[1]);

    ndarray img(2, size);

    for(int i=0;i<img.d_size;++i){
        img.data[i] = read_num();
    }
    return img;
}

void show_img(ndarray &x){
    Serial.print("x:");
    Serial.print(x.shape[0]);
    Serial.print(" y:");
    Serial.println(x.shape[1]);
    for(int i=0;i<x.shape[0];++i){
        for(int j = 0;j < x.shape[1]; ++j){
            Serial.print(x.data[i*x.shape[1]+j], DEC);
            Serial.print("\t");
        }
        Serial.print("\n");
    }
}

struct timespec start_, end_;
void start(){
    clock_gettime(CLOCK_REALTIME, &start_);
}
void end(){
    clock_gettime(CLOCK_REALTIME, &end_);
    int t_ns = (end_.tv_nsec - start_.tv_nsec)/1000000;
    int t_s = end_.tv_sec - start_.tv_sec;
    if(t_ns<0){
        t_s -= 1;
        t_ns += 1000;
    }
    Serial.print("elapsed time:");
    Serial.print(t_s);
    Serial.print(".");
    Serial.print(t_ns);
    Serial.println("sec");
}

void LassoADMM(){
    Serial.println("-x-x-x-x-x-x-");
    ndarray d = load_img();
    ndarray y = load_img();
    admm a(y, d, 0.3, 1.0);
    a.init();
    a.train(1000);
    ndarray x = a.get_sparse_vec();
    show_img(x);
    Serial.println("-x-x-x-x-x-x-");
}

void FusedLasso(){
    ndarray f = load_img();
    int shape[] = {f.shape[0], f.shape[0]};
    ndarray k(2, shape);
    k.add_diag_mat(1);
    fused_lasso b(f, k, 0.02, 0.2, 0, 0.8);
    b.init();
    b.train(100);
    ndarray z = b.get_sparse_vec();
    show_img(z);
    delay(1000);    
    Serial.println("-x-x-x-x-x-x-");
}

void FFT(){
    ndarray x = load_img();
    complex *x_ = (complex *)calloc(x.d_size, sizeof(complex));
    for (int i = 0; i < x.d_size; ++i){
        x_[i].real = x.data[i];
        x_[i].imag = 0;
    }
    complex *freq = fft(x_, x.d_size);
    for (int i = 0; i < x.d_size; ++i)
    {
        Serial.print(freq[i].real);
        Serial.print("\t");
        Serial.print(freq[i].imag);
        Serial.println("j");
    }
}

void IFFT(){
    ndarray x = load_img();
    complex *x_ = (complex *)calloc(x.d_size, sizeof(complex));
    for (int i = 0; i < x.d_size; ++i)
    {
        x_[i].real = x.data[i];
        x_[i].imag = 0;
    }
    complex *freq = fft(x_, x.d_size);
    complex *time = ifft(freq, x.d_size);
    for (int i = 0; i < x.d_size; ++i)
    {
        Serial.print(time[i].real);
        Serial.print("\t");
        Serial.print(time[i].imag);
        Serial.println("j");
    }
}

void setup() {
    Serial.begin(115200);

    // int N = read_int();
    // int bit = bit_count(N - 1);
    // for (int i = 0; i < N; ++i)
    // {
    //     Serial.println(bit_reverse(i, bit));
    // }
    // FusedLasso();
    IFFT();
    Serial.println("-x-x-x-x-x-x-");
}

void loop() {

}
