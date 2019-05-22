#include "lasso_ndarray.h"
#include "lasso_xorshift.h"
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
// #include <sdk/config.h>
// #include <stdio.h>
// #include <arch/board/board.h>
#include <pthread.h>

// #include <math.h>

ndarray::ndarray(){
    this->dim=0;
    this->d_size=0;
    this->shape=NULL;
    this->data=NULL;
}

ndarray::ndarray(int dim, int *shape){
    int d=0;
    this->shape = (int*)calloc(dim, sizeof(int));
    for(int i =0;i<dim;++i){
        this->shape[i]=shape[i];
    }
    this->dim = dim;
    this->d_size = this->size();
    this->data = (float*)calloc(this->d_size, sizeof(float));
}


ndarray::ndarray(float* ptr, int dim, int *shape){
    this->data = ptr;
    this->dim = dim;
    this->shape= shape;
    this->d_size = this->size();
}

ndarray::~ndarray(){
    if(this->dim){
        this->destroy();
    }
}

void ndarray::destroy(){
    free(this->data);
    free(this->shape);
}

ndarray ndarray::operator[](int x){
    int p=0;
    for(int i = 1; i<this->dim;++i){
        p*=shape[i];
    }
    return ndarray(this->data+p*x, this->dim-1, shape+1);
}

void ndarray::operator<=(ndarray &x){
    this->destroy();
    this->dim=x.dim;
    this->shape=x.shape;
    this->d_size=x.d_size;
    this->data=x.data;
    x.dim = 0;
}

ndarray ndarray::operator+(ndarray &x){
    ndarray y = this->make_similar_dim_ndarray();
    for(int i = 0;i < this->d_size; ++i){
        y.data[i] = this->data[i] + x.data[i];
    }
    return y;
}
ndarray ndarray::operator-(ndarray &x){
    ndarray y = this->make_similar_dim_ndarray();
    for(int i = 0;i < this->d_size; ++i){
        y.data[i] = this->data[i] - x.data[i];
    }
    return y;
}
ndarray ndarray::operator*(ndarray &x){
    ndarray y = this->make_similar_dim_ndarray();
    for(int i = 0;i < this->d_size; ++i){
        y.data[i] = this->data[i] * x.data[i];
    }
    return y;
}
ndarray ndarray::operator/(ndarray &x){
    ndarray y = this->make_similar_dim_ndarray();
    for(int i = 0;i < this->d_size; ++i){
        y.data[i] = this->data[i] / x.data[i];
    }
    return y;
}

void ndarray::operator+=(ndarray &x) {
    for(int i = 0;i<this->d_size;++i){
        this->data[i] += x.data[i];
    }
}
void ndarray::operator-=(ndarray &x) {
    for(int i = 0;i<this->d_size;++i){
        this->data[i] -= x.data[i];
    }
}
void ndarray::operator*=(ndarray &x) {
    for(int i = 0;i<this->d_size;++i){
        this->data[i] *= x.data[i];
    }
}
void ndarray::operator/=(ndarray &x) {
    for(int i = 0;i<this->d_size;++i){
        this->data[i] /= x.data[i];
    }
}


ndarray ndarray::make_similar_dim_ndarray(){
    ndarray x(this->dim, this->shape);
    return x;
}

void ndarray::operator=(float x){
    for(int i = 0;i<this->d_size;++i){
        this->data[i]=x;
    }
}


void ndarray::add_diag_mat(float x){
    for(int i = 0;i < (this->shape[0] < this->shape[1] ? this->shape[0] :this->shape[1]) ;++i){
        this->data[i*this->shape[1]+i] += x;
    }
}


void ndarray::operator+=(float x) {
    for(int i = 0;i<this->d_size;++i){
        this->data[i]+=x;
    }
}


void ndarray::operator-=(float x) {
    for(int i = 0;i<this->d_size;++i){
        this->data[i]-=x;
    }
}


void ndarray::operator*=(float x) {
    for(int i = 0;i<this->d_size;++i){
        this->data[i]*=x;
    }
}


void ndarray::operator/=(float x) {
    for(int i = 0;i<this->d_size;++i){
        this->data[i]/=x;
    }
}


ndarray ndarray::operator+(float x) {
    ndarray a = this->copy();
    a+=x;
    return a;
}


ndarray ndarray::operator-(float x) {
    ndarray a = this->copy();
    a-=x;
    return a;
}


ndarray ndarray::operator*(float x) {
    ndarray a = this->copy();
    a*=x;
    return a;
}


ndarray ndarray::operator/(float x) {
    ndarray a = this->copy();
    a/=x;
    return a;
}


int ndarray::size(){
    int x=1;
    for(int i=0;i<this->dim;++i){
        x*=shape[i];
    }
    return x;
}


ndarray ndarray::copy(){
    float *ptr = (float*)calloc(this->d_size, sizeof(float));
    memcpy(ptr, this->data, this->d_size*sizeof(float));
    int *shape = (int*)calloc(this->dim, sizeof(int));
    memcpy(shape, this->shape, this->dim*sizeof(int));
    ndarray a = ndarray(ptr, this->dim, shape);
    return a;
}

ndarray dot(ndarray &a, ndarray &b, int a_T, int b_T){
    int i_ = a_T ? a.shape[1] : a.shape[0];
    int j_ = b_T ? b.shape[0] : b.shape[1];
    int k_ = a_T ? a.shape[0] : a.shape[1];

    int shape[] = {i_, j_};
    ndarray x(2, shape);
    
    for(int i = 0; i < i_; ++i)
    {
        for(int j = 0; j < j_; ++j)
        {
            double r = 0;
            for(int k = 0; k < k_; ++k)
            {
                r += a.data[a_T ? k*i_+i : i*k_+k] * b.data[b_T ? j*k_+k : k*j_+j];
            }
            x.data[i*j_+j] = r;
        }
    }
    return x;
}


typedef struct dot_data {
    ndarray* a;
    ndarray* b;
    ndarray* x;
    int s;
    int t;
    int i_, j_, k_;
    int a_T, b_T;
};
// void dot_(ndarray *a, ndarray *b, ndarray *x, int s, int t, int i_ , int j_, int k_, int a_T, int b_T){
void dot_(dot_data *data){
    for(int i = data->s; i < data->t; ++i)
    {
        for(int j = 0; j < data->j_; ++j)
        {
            double r = 0;
            for(int k = 0; k < data->k_; ++k)
            {
                r += data->a->data[data->a_T ? k*data->i_+i : i*data->k_+k] * data->b->data[data->b_T ? j*data->k_+k : k*data->j_+j];
            }
            data->x->data[i*data->j_+j] = r;
        }
    }
}
ndarray parallel_dot(ndarray &a, ndarray &b, int a_T, int b_T){
    int i_ = a_T ? a.shape[1] : a.shape[0];
    int j_ = b_T ? b.shape[0] : b.shape[1];
    int k_ = a_T ? a.shape[0] : a.shape[1];

    int shape[] = {i_, j_};
    ndarray x(2, shape);

    pthread_t thread[6];
    dot_data data;
    data.a = &a;
    data.b = &b;
    data.x = &x;
    data.t = 0;
    data.i_ = i_;
    data.j_ = j_;
    data.k_ = k_;
    data.a_T = a_T;
    data.b_T = b_T;
    for(int i = 0; i < i_%6; ++i){
        data.s = data.t;
        data.t += i_/6+1;
        pthread_create(&thread[i], NULL, dot_, (void*)&data);
    }
    for(int i = i_%6; i < 6; ++i){
        data.s = data.t;
        data.t += i_/6;
        pthread_create(&thread[i], NULL, dot_, (void*)&data);

    }

    for(int i = 0; i < 6; ++i){
        pthread_join(thread[i], NULL);
    }

    return x;
}

ndarray inv(ndarray &x_){
    int n = x_.shape[0];
    int shape[] = {n, n};
    ndarray x = x_.copy();
    ndarray a(2, shape);
    for(int i = 0; i < n; ++i){
        a.data[i*n+i] = 1;
    }

    for(int i = 0; i < n; ++i){
        float t = x.data[i*n+i];
        for(int j = 0; j < n; ++j){
            x.data[i*n+j] /= t;
            a.data[i*n+j] /= t;
        }
        for(int j = 0; j < n; ++j){
            if(i != j){
                float t = x.data[j*n+i];
                for(int k = 0; k < n; ++k){
                    x.data[j*n+k] -= x.data[i*n+k]*t;
                    a.data[j*n+k] -= a.data[i*n+k]*t;
                }
            }
        }
    }
    // x.destroy();
    return a;
}


float ndarray::sum(){
    float x = 0;
    for(int i = 0;i<this->d_size;++i){
        x+=data[i];
    }
    return x;
}


void random(ndarray &x, float a, float b) {
    float range = b-a;
    for(int i = 0; i < x.d_size; ++i){
        x.data[i] = my_rand(range) + a;
    }
}

void arange(ndarray &x) {
    for(int i = 0; i < x.d_size; ++i){
        x.data[i] = i;
    }
}
