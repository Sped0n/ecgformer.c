#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "components.h"
#include "smlcomp.h"
#include "smlio.h"

const char* fmtdbg = "%8" SML_PRTREAL "g ";

void
Embedding(Tensor i, Tensor o, const struct EmbeddingParamsPath params)
{
  /*
   * SEQ_LEN x 1 -> SEQ_LEN x EMBED_DIM
   */
  Tensor tmp =
    MatDim(1, EMBED_DIM); // Weight and bias have the same shape 1xEMBED_DIM

  MatReadFile(params.linear_weight, tmp);
  MatMul(o, i, tmp);
  MatReadFile(params.linear_bias, tmp);
  MatAddBias(tmp, o);

  MatUnDim(tmp);

  MatLayerNorm(o, 1, MatCols(o));

  MatRelu(o);
}

void
MultiHeadAttention(Tensor i, const struct MultHeadAttnParamsPath params)
{
  /*
   * SEQ_LEN x EMBED_DIM -> SEQ_LEN x EMBED_DIM
   */

  // Residual
  Tensor tmp = MatDim(MatRows(i), MatCols(i));
  MatCopy(tmp, i);

  // LayerNorm
  MatLayerNorm(tmp, 1, MatCols(tmp));

  // Query, Key, Value projection
  Tensor q_proj_result = MatDim(SEQ_LEN, EMBED_DIM);
  Tensor q_proj_weight = MatDim(EMBED_DIM, EMBED_DIM);
  MatReadFile(params.q_proj_weight, q_proj_weight);
  MatMul(q_proj_result, tmp, q_proj_weight);
  MatUnDim(q_proj_weight);
  Tensor q_proj_bias = MatDim(1, EMBED_DIM);
  MatReadFile(params.q_proj_bias, q_proj_bias);
  MatAddBias(q_proj_bias, q_proj_result);
  MatUnDim(q_proj_bias);

  Tensor k_proj_result = MatDim(SEQ_LEN, EMBED_DIM);
  Tensor k_proj_weight = MatDim(EMBED_DIM, EMBED_DIM);
  MatReadFile(params.k_proj_weight, k_proj_weight);
  MatMul(k_proj_result, tmp, k_proj_weight);
  MatUnDim(k_proj_weight);
  Tensor k_proj_bias = MatDim(1, EMBED_DIM);
  MatReadFile(params.k_proj_bias, k_proj_bias);
  MatAddBias(k_proj_bias, k_proj_result);
  MatUnDim(k_proj_bias);
  Tensor k_proj_result_t =
    MatDim(MatCols(k_proj_result), MatRows(k_proj_result));
  MatTran(k_proj_result_t, k_proj_result);
  MatUnDim(k_proj_result);

  Tensor v_proj_result = MatDim(SEQ_LEN, EMBED_DIM);
  Tensor v_proj_weight = MatDim(EMBED_DIM, EMBED_DIM);
  MatReadFile(params.v_proj_weight, v_proj_weight);
  MatMul(v_proj_result, tmp, v_proj_weight);
  MatUnDim(v_proj_weight);
  Tensor v_proj_bias = MatDim(1, EMBED_DIM);
  MatReadFile(params.v_proj_bias, v_proj_bias);
  MatAddBias(v_proj_bias, v_proj_result);
  MatUnDim(v_proj_bias);

  MatUnDim(tmp);

  // Reshape to NUM_HEAD chunks
  Tensor* q_heads = malloc(NUM_HEADS * sizeof(Tensor));
  Tensor* k_heads = malloc(NUM_HEADS * sizeof(Tensor));
  Tensor* v_heads = malloc(NUM_HEADS * sizeof(Tensor));
  size_t head_size = EMBED_DIM / NUM_HEADS;
  for (size_t h = 0; h < NUM_HEADS; h++) {
    q_heads[h] = MatDim(SEQ_LEN, head_size);
    k_heads[h] = MatDim(head_size, SEQ_LEN); // k is already transposed
    v_heads[h] = MatDim(SEQ_LEN, head_size);
    for (size_t row = 0; row < SEQ_LEN; row++) {
      for (size_t col = 0; col < head_size; col++) {
        MatSet0(q_heads[h],
                row,
                col,
                MatGet0(q_proj_result, row, h * head_size + col));
        MatSet0(v_heads[h],
                row,
                col,
                MatGet0(v_proj_result, row, h * head_size + col));

        MatSet0(k_heads[h],
                col,
                row,
                MatGet0(k_proj_result_t, h * head_size + col, row));
      }
    }
  }
  MatUnDim(q_proj_result);
  MatUnDim(k_proj_result_t);
  MatUnDim(v_proj_result);

  // Scaled Dot-Product Attention
  Tensor* qkv_outputs = malloc(NUM_HEADS * sizeof(Tensor));
  for (size_t head_index = 0; head_index < NUM_HEADS; ++head_index) {
    Tensor attn_nom = MatDim(SEQ_LEN, SEQ_LEN);
    MatMul(attn_nom, q_heads[head_index], k_heads[head_index]);
    MatMulScalar(attn_nom, attn_nom, 1.0 / SQRT_DK);
    MatSoftmax(attn_nom);
    MatUnDim(q_heads[head_index]);
    MatUnDim(k_heads[head_index]);

    qkv_outputs[head_index] = MatDim(SEQ_LEN, EMBED_DIM / NUM_HEADS);
    MatMul(qkv_outputs[head_index], attn_nom, v_heads[head_index]);
    MatUnDim(attn_nom);
    MatUnDim(v_heads[head_index]);
  }
  free(q_heads);
  free(k_heads);
  free(v_heads);

  // Concatenate heads
  tmp = MatDim(SEQ_LEN, EMBED_DIM);
  for (size_t row_index = 0; row_index < SEQ_LEN; ++row_index) {
    for (size_t col_index = 0; col_index < EMBED_DIM; ++col_index) {
      size_t head_index = col_index / NUM_HEADS;
      size_t inner_index = col_index % NUM_HEADS;
      MatSet0(tmp,
              row_index,
              col_index,
              MatGet0(qkv_outputs[head_index], row_index, inner_index));
    }
  }
  for (size_t head_index = 0; head_index < NUM_HEADS; ++head_index) {
    MatUnDim(qkv_outputs[head_index]);
  }
  free(qkv_outputs);

  // Final projection
  Tensor final_proj_weight = MatDim(EMBED_DIM, EMBED_DIM);
  MatReadFile(params.final_proj_weight, final_proj_weight);
  MatMul(tmp, tmp, final_proj_weight);
  MatUnDim(final_proj_weight);
  Tensor final_proj_bias = MatDim(1, EMBED_DIM);
  MatReadFile(params.final_proj_bias, final_proj_bias);
  MatAddBias(final_proj_bias, tmp);
  MatUnDim(final_proj_bias);

  // Residual connection
  MatAdd(i, i, tmp);
  MatUnDim(tmp);
}

void
MultiLayerPerceptron(Tensor i, struct MLPParamsPath params)
{
  /*
   * SEQ_LEN x EMBED_DIM -> SEQ_LEN x EMBED_DIM
   * Note: no up projection here
   */

  // Residual
  Tensor tmp = MatDim(MatRows(i), MatCols(i));
  MatCopy(tmp, i);

  // LayerNorm
  MatLayerNorm(tmp, 1, MatCols(tmp));

  // Feed Forward
  Tensor ff1_weight = MatDim(EMBED_DIM, EMBED_DIM);
  MatReadFile(params.linear_1_weight, ff1_weight);
  MatMul(tmp, tmp, ff1_weight);
  MatUnDim(ff1_weight);
  Tensor ff1_bias = MatDim(1, EMBED_DIM);
  MatReadFile(params.linear_1_bias, ff1_bias);
  MatAddBias(ff1_bias, tmp);
  MatUnDim(ff1_bias);

  // ReLU
  MatRelu(tmp);

  // Feed Forward
  Tensor ff2_weight = MatDim(EMBED_DIM, EMBED_DIM);
  MatReadFile(params.linear_2_weight, ff2_weight);
  MatMul(tmp, tmp, ff2_weight);
  Tensor ff2_bias = MatDim(1, EMBED_DIM);
  MatReadFile(params.linear_2_bias, ff2_bias);
  MatAddBias(ff2_bias, tmp);
  MatUnDim(ff2_bias);

  // Residual connection
  MatAdd(i, i, tmp);
  MatUnDim(tmp);
}

void
Classifier(Tensor i, Tensor o, struct ClassifierParamsPath params)
{
  /*
   * SEQ_LEN x EMBED_DIM -> 1 x NUM_CLASSES
   */

  // ReduceMean along SEQ_LEN
  Tensor tmp = MatDim(1, EMBED_DIM);
  MatColSum(tmp, i);

  // LayerNorm
  MatLayerNorm(tmp, 1, MatCols(tmp));

  // Feed Forward
  Tensor ff_weight = MatDim(EMBED_DIM, NUM_CLASSES);
  MatReadFile(params.linear_weight, ff_weight);
  MatMul(o, tmp, ff_weight);
  MatUnDim(tmp);
  MatUnDim(ff_weight);
  Tensor ff_bias = MatDim(1, NUM_CLASSES);
  MatReadFile(params.linear_bias, ff_bias);
  MatAddBias(ff_bias, o);
  MatUnDim(ff_bias);
}
