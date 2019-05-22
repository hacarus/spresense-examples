class ndarray{
    public:
    ndarray();
    ndarray(int dim, int *shape);
    ndarray(float* ptr, int dim, int *shape);
    ndarray make_similar_dim_ndarray();
    ~ndarray();
    void destroy();
    float *data;
    int *shape;
    int d_size, dim;
    ndarray operator[](int x);
    // void operator=(ndarray &x);
    void operator<=(ndarray &x);
    void operator=(float x);
    void operator+=(float x);
    void add_diag_mat(float x);
    void operator-=(float x);
    void operator*=(float x);
    void operator/=(float x);
    void operator+=(ndarray &x);
    void operator-=(ndarray &x);
    void operator*=(ndarray &x);
    void operator/=(ndarray &x);
    ndarray operator+(float x);
    ndarray operator-(float x);
    ndarray operator*(float x);
    ndarray operator/(float x);
    ndarray operator+(ndarray &x);
    ndarray operator-(ndarray &x);
    ndarray operator*(ndarray &x);
    ndarray operator/(ndarray &x);
    // ndarray sqrt();
    int size();
    ndarray copy();
    float sum();
    int dot(ndarray x);

};

// ndarray dot(ndarray a, ndarray b);
ndarray dot(ndarray &a, ndarray &b, int a_T = false, int b_T = false);
ndarray parallel_dot(ndarray &a, ndarray &b, int a_T = false, int b_T = false);
ndarray inv(ndarray &x_);
void random(ndarray &x, float a, float b);
void arange(ndarray &x);
