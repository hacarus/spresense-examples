ndarray make_patch(ndarray x, int patch_size);



class admm
{
private:
    // ndarray *u;
    // ndarray Z;
    // ndarray x;
    ndarray inv_;
    ndarray w_t, z_t, h_t;
    float thresh;
    ndarray inv_matrix_XTy, inv_matrix_DT;
    // ndarray normalize(ndarray x);
public:
    // admm(ndarray &D, ndarray &X, ndarray &Y, float lambda, float rho);
    ndarray D;
    admm(ndarray &Y, ndarray &X, float lambda, float rho);
    ndarray X;
    ndarray Y;
    ndarray debug;
    float lambda, rho;
    void fit();
    void init();
    virtual void generate_transform_matrix(int n_features);
    ndarray soft_thresholding(ndarray &x);
    ndarray hard_thresholding(ndarray &x);
    void train(int iter);
    ndarray get_sparse_vec();
    // ~admm();
};

class fused_lasso : public admm
{
private:
    /* data */
public:
    fused_lasso(ndarray &Y, ndarray &X, float lambda, float rho, float sparse_coef=1.0, float fused_coef=1.0);
    void generate_transform_matrix(int n_features);
    float sparse_coef, fused_coef;
    // ~fused_lasso();
};


