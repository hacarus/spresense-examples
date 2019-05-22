#include "lasso_ndarray.h"
#include "lasso_train.h"

ndarray make_patch(ndarray x, int patch_size){
    int n = x.shape[0]/patch_size;
    int m = x.shape[1]/patch_size;
    int shape[] = {patch_size*patch_size, n*m};
    ndarray a(2, shape);
    for(int i=0;i<n;++i){
        for(int j = 0; j < m; ++j)
        {
            for(int k = 0; k < patch_size; ++k)
            {
                for(int l = 0; l < patch_size; ++l)
                {
                    a.data[a.shape[1]*(k*patch_size+l)+i*m+j]=x.data[(i*patch_size+k)*x.shape[1]+j*patch_size+l];
                }
            }
        }
    }
    return a;
}

admm::admm(ndarray &Y, ndarray &X, float lambda, float rho)
{
    ndarray y = Y.copy();
    this->Y <= y;
    ndarray d = X.copy();
    this->X <= d;
    // this->X = X;
    this->lambda = lambda;
    this->rho = rho;
    this->thresh = lambda/rho;
    // this->init();

}

void admm::generate_transform_matrix(int n_features){
    int shape[] = {n_features, n_features};
    ndarray d(2, shape);
    d.add_diag_mat(1);
    this->D <= d;
}


void admm::init(){
    this->generate_transform_matrix(X.shape[1]);

    ndarray hoge = dot(this->X, this->X, true);
    hoge /= this->X.shape[0];
    ndarray fuga = dot(this->D, this->D, true);
    fuga *= rho;
    hoge += fuga;
    ndarray inv_matrix = inv(hoge);
    ndarray piyo = dot(inv_matrix, X, false, true);

    ndarray inv_matrix_XTy = dot(piyo, this->Y);
    inv_matrix_XTy /= this->X.shape[0];

    ndarray foo = this->D*rho;
    ndarray inv_matrix_DT = dot(inv_matrix, foo, false, true);

    // this->inv_matrix <= inv_matrix;
    this->inv_matrix_XTy <= inv_matrix_XTy;
    this->inv_matrix_DT <= inv_matrix_DT;

    // // hoge.add_diag_mat(this->rho);   
    // // // this->debug <= hoge;
    // // ndarray fuga = inv(hoge);
    // // this->inv_ <= fuga;
    // ndarray x = dot(this->X, this->Y, true);
    // // ndarray d = x.copy();
    // // this->debug <= d;
    // // // this->debug <= x;
    // x /= this->X.shape[0];
    // this->w_t <= x;
    // ndarray x_ = this->w_t.copy();;
    // this->x <= x_;
    // ndarray z_ = this->w_t.copy();
    // ndarray z = dot(this->D, z_);
    // this->z_t <= z;
    // ndarray y = this->z_t.make_similar_dim_ndarray();
    // this->h_t <= y;


    ndarray moge = dot(this->D, this->D, true);
    moge /= this->D.shape[0];       
    moge.add_diag_mat(this->rho);
    // ndarray fuga = inv(moge);
    // this->inv_ <= fuga;
    ndarray x = dot(this->D, this->Y, true);
    ndarray d = x.copy();
    this->debug <= d;
    x /= this->D.shape[0];
    this->w_t <= x;
    ndarray z = this->w_t.copy();
    this->z_t <= z;
    ndarray y = this->z_t.make_similar_dim_ndarray();
    this->h_t <= y;
}

ndarray admm::soft_thresholding(ndarray &x){
    ndarray y(x.dim, x.shape);
    for(int i = 0;i < x.d_size;++i){
        y.data[i] = x.data[i] > this->thresh ? x.data[i] - this->thresh :
               x.data[i] < -this->thresh ? x.data[i] + this->thresh : 0.0;
    }
    return y;
}

ndarray admm::hard_thresholding(ndarray &x){
    ndarray y(x.dim, x.shape);
    for(int i = 0;i < x.d_size;++i){
        y.data[i] = x.data[i] > this->thresh || x.data[i] < -this->thresh ? x.data[i] : 0.0;
    }
    return y;
}

void admm::fit(){


    // w_t[:, k] = inv_matrix_XTy[:, k] + inv_matrix_DT.dot(z_t[:, k] - h_t[:, k] / rho)
    // Dw_t = D.dot(w_t[:, k])
    // z_t[:, k] = _soft_threshold(Dw_t + h_t[:, k] / rho, threshold)
    // h_t[:, k] += rho * (Dw_t - z_t[:, k])

    // x = np.dot(inv_matrix, np.dot(A.T, b) / N + self.rho * z - y)
    // z = self._soft_threshold(x + y / self.rho)
    // y += self.rho * (x - z)




    ndarray u = this->h_t / this->rho;
    ndarray t = this->z_t - u;
    ndarray v = dot(inv_matrix_DT, t);
    ndarray w_t = inv_matrix_XTy + v;

    ndarray Dw_t = dot(this->D, w_t);
    u += Dw_t;
    ndarray z_t = this->soft_thresholding(u);

    ndarray w = Dw_t - z_t;
    w *= this->rho;

    this->w_t <= w_t;
    this->z_t <= z_t;
    h_t += w;



    // t += this->x;
    // t -= this->h_t;
    // this->debug <= t;
    // ndarray x = dot(this->inv_, t);
    // u += x;
    // ndarray z = this->soft_thresholding(u);
    // ndarray v = (x - z);
    // v *= this->rho;
    // this->h_t += v;

}

void admm::train(int iter){
    for(int i=0;i<iter;++i){
        this->fit();
    }
}

ndarray admm::get_sparse_vec(){
    return this->w_t;
}


fused_lasso::fused_lasso(ndarray &Y, ndarray &X, float lambda, float rho, float sparse_coef, float fused_coef) : admm(Y, X, lambda, rho)
{
    this->sparse_coef = sparse_coef;
    this->fused_coef = fused_coef;
}

void fused_lasso::generate_transform_matrix(int n_features){
    int shape[] = {n_features, n_features};
    ndarray d(2, shape);
    d.data[0] = this->sparse_coef;
    for(int i = 1; i < n_features; ++i){
        d.data[i*n_features+i-1] = -this->fused_coef;
        d.data[i*n_features+i] =    this->fused_coef + this->sparse_coef;
    }
    this->D <= d;
}


// admm::~admm()
// {
// }
