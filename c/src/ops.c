#include <math.h>
#include <stddef.h>

#include "main.h"

void
MatSoftmax(Tensor i)
{
  size_t rows = MatRows(i), cols = MatCols(i);

  float largest = MatMax(i);

  for (size_t row_index = 0; row_index < rows; ++row_index) {
    float sum_exp = 0.;
    for (size_t col_index = 0; col_index < cols; ++col_index) {
      float e = sml_exp(MatGet0(i, row_index, col_index) - largest);
      sum_exp += e;
      MatSet0(i, row_index, col_index, e);
    }
    for (size_t col_index = 0; col_index < cols; ++col_index) {
      float e = MatGet0(i, row_index, col_index) / sum_exp;
      MatSet0(i, row_index, col_index, e);
    }
  }
}

void
MatLayerNorm(Tensor i, size_t normalized_shape_row, size_t normalized_shape_col)
{
  size_t rows = MatRows(i), cols = MatCols(i);

  for (size_t row_index = 0; row_index < rows;
       row_index += normalized_shape_row) {
    for (size_t col_index = 0; col_index < cols;
         col_index += normalized_shape_col) {
      // Calculate mean
      float mean = 0.0f;
      for (size_t block_row_index = 0; block_row_index < normalized_shape_row;
           ++block_row_index) {
        for (size_t block_col_index = 0; block_col_index < normalized_shape_col;
             ++block_col_index) {
          mean += MatGet0(
            i, row_index + block_row_index, col_index + block_col_index);
        }
      }
      mean /= (normalized_shape_row * normalized_shape_col);

      // Calculate variance
      float variance = 0.0f;
      for (size_t block_row_index = 0; block_row_index < normalized_shape_row;
           ++block_row_index) {
        for (size_t block_col_index = 0; block_col_index < normalized_shape_col;
             ++block_col_index) {
          float diff = MatGet0(i,
                               row_index + block_row_index,
                               col_index + block_col_index) -
                       mean;
          variance += diff * diff;
        }
      }
      variance /= (normalized_shape_row * normalized_shape_col);

      // Normalize
      float std_dev =
        sqrtf(variance + 1e-5f); // Adding epsilon for numerical stability
      for (size_t block_row_index = 0; block_row_index < normalized_shape_row;
           ++block_row_index) {
        for (size_t block_col_index = 0; block_col_index < normalized_shape_col;
             ++block_col_index) {
          float normalized_value =
            (MatGet0(
               i, row_index + block_row_index, col_index + block_col_index) -
             mean) /
            std_dev;
          MatSet0(i,
                  row_index + block_row_index,
                  col_index + block_col_index,
                  normalized_value);
        }
      }
    }
  }
}

void
MatAddBias(Tensor i_bias, Tensor i)
{
  size_t rows = MatRows(i), cols = MatCols(i);

  for (size_t row_index = 0; row_index != rows; ++row_index) {
    for (size_t col_index = 0; col_index != cols; ++col_index) {
      MatSet0(i,
              row_index,
              col_index,
              MatGet0(i, row_index, col_index) + MatGet0(i_bias, 0, col_index));
    }
  }
}

void
MatRelu(Tensor i)
{
  size_t rows = MatRows(i), cols = MatCols(i);

  for (size_t row_index = 0; row_index < rows; ++row_index) {
    for (size_t col_index = 0; col_index < cols; ++col_index) {
      float val = MatGet0(i, row_index, col_index);
      // ReLU: max(0, x)
      MatSet0(i, row_index, col_index, val > 0 ? val : 0);
    }
  }
}
