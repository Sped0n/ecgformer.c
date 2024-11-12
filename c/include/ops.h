#include <sml.h>

typedef MATRIX Tensor;

void
MatSoftmax(Tensor i);

void
MatLayerNorm(Tensor i,
             size_t normalized_shape_row,
             size_t normalized_shape_col);

void
MatAddBias(Tensor i_bias, Tensor i);

void
MatRelu(Tensor i);

void
MatTranspose(Tensor i);
