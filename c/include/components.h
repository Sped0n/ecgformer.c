#include "ops.h"

#define NUM_HEADS 4
#define EMBED_DIM 16
#define SEQ_LEN 300
#define SQRT_DK 2 // sqrt(EMBED_DIM / NUM_HEADS)
#define NUM_CLASSES 5

struct EmbeddingParamsPath
{
  const char* linear_weight;
  const char* linear_bias;
};

struct MultHeadAttnParamsPath
{
  const char* q_proj_weight;
  const char* q_proj_bias;
  const char* k_proj_weight;
  const char* k_proj_bias;
  const char* v_proj_weight;
  const char* v_proj_bias;
  const char* final_proj_weight;
  const char* final_proj_bias;
};

struct MLPParamsPath
{
  const char* linear_1_weight;
  const char* linear_1_bias;
  const char* linear_2_weight;
  const char* linear_2_bias;
};

struct ClassifierParamsPath
{
  const char* linear_weight;
  const char* linear_bias;
};

struct ParamsPath
{
  const struct EmbeddingParamsPath embedding;
  const struct MultHeadAttnParamsPath mha_0;
  const struct MLPParamsPath mlp_0;
  const struct MultHeadAttnParamsPath mha_1;
  const struct MLPParamsPath mlp_1;
  const struct ClassifierParamsPath classifier;
};

void
Embedding(Tensor i, Tensor o, const struct EmbeddingParamsPath params);

void
MultiHeadAttention(Tensor i, const struct MultHeadAttnParamsPath params);

void
MultiLayerPerceptron(Tensor i, struct MLPParamsPath params);

void
Classifier(Tensor i, Tensor o, struct ClassifierParamsPath params);
