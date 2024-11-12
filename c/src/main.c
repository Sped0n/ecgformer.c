#include <stdlib.h>
#include <string.h>

#include "main.h"

const char* fmt = "%8" SML_PRTREAL "g ";

struct ParamsPath
InitParamPath(void)
{
  const struct EmbeddingParamsPath embedding = {
    .linear_weight = "../../assets/params/embedding.linear.weight.txt",
    .linear_bias = "../../assets/params/embedding.linear.bias.txt"
  };
  const struct MultHeadAttnParamsPath mha_0 = {
    .q_proj_weight = "../../assets/params/"
                     "encoder_layers.0.attention.queries_projection.weight.txt",
    .q_proj_bias = "../../assets/params/"
                   "encoder_layers.0.attention.queries_projection.bias.txt",
    .k_proj_weight = "../../assets/params/"
                     "encoder_layers.0.attention.keys_projection.weight.txt",
    .k_proj_bias =
      "../../assets/params/encoder_layers.0.attention.keys_projection.bias.txt",
    .v_proj_weight = "../../assets/params/"
                     "encoder_layers.0.attention.values_projection.weight.txt",
    .v_proj_bias = "../../assets/params/"
                   "encoder_layers.0.attention.values_projection.bias.txt",
    .final_proj_weight =
      "../../assets/params/"
      "encoder_layers.0.attention.final_projection.weight.txt",
    .final_proj_bias =
      "../../assets/params/encoder_layers.0.attention.final_projection.bias.txt"
  };
  const struct MLPParamsPath mlp_0 = {
    .linear_1_weight =
      "../../assets/params/encoder_layers.0.mlp.linear1.weight.txt",
    .linear_1_bias =
      "../../assets/params/encoder_layers.0.mlp.linear1.bias.txt",
    .linear_2_weight =
      "../../assets/params/encoder_layers.0.mlp.linear2.weight.txt",
    .linear_2_bias = "../../assets/params/encoder_layers.0.mlp.linear2.bias.txt"
  };
  const struct MultHeadAttnParamsPath mha_1 = {
    .q_proj_weight = "../../assets/params/"
                     "encoder_layers.1.attention.queries_projection.weight.txt",
    .q_proj_bias = "../../assets/params/"
                   "encoder_layers.1.attention.queries_projection.bias.txt",
    .k_proj_weight = "../../assets/params/"
                     "encoder_layers.1.attention.keys_projection.weight.txt",
    .k_proj_bias =
      "../../assets/params/encoder_layers.1.attention.keys_projection.bias.txt",
    .v_proj_weight = "../../assets/params/"
                     "encoder_layers.1.attention.values_projection.weight.txt",
    .v_proj_bias = "../../assets/params/"
                   "encoder_layers.1.attention.values_projection.bias.txt",
    .final_proj_weight =
      "../../assets/params/"
      "encoder_layers.1.attention.final_projection.weight.txt",
    .final_proj_bias =
      "../../assets/params/encoder_layers.1.attention.final_projection.bias.txt"
  };
  const struct MLPParamsPath mlp_1 = {
    .linear_1_weight =
      "../../assets/params/encoder_layers.1.mlp.linear1.weight.txt",
    .linear_1_bias =
      "../../assets/params/encoder_layers.1.mlp.linear1.bias.txt",
    .linear_2_weight =
      "../../assets/params/encoder_layers.1.mlp.linear2.weight.txt",
    .linear_2_bias = "../../assets/params/encoder_layers.1.mlp.linear2.bias.txt"
  };
  const struct ClassifierParamsPath classifier = {
    .linear_weight = "../../assets/params/classifier.linear.weight.txt",
    .linear_bias = "../../assets/params/classifier.linear.bias.txt"
  };

  const struct ParamsPath result = { .embedding = embedding,
                                     .mha_0 = mha_0,
                                     .mlp_0 = mlp_0,
                                     .mha_1 = mha_1,
                                     .mlp_1 = mlp_1,
                                     .classifier = classifier };

  return result;
}

int
main(void)
{
  const struct ParamsPath params_path = InitParamPath();

  Tensor input = MatDim(300, 1);
  MatReadFile("../../assets/input.txt", input);

  Tensor embed_input = MatDim(300, 16);
  Embedding(input, embed_input, params_path.embedding);
  MatUnDim(input);
  MatWriteFile("embed_output.txt", embed_input, fmt, "Embed: ");

  MultiHeadAttention(embed_input, params_path.mha_0);
  MatWriteFile("mha_0_output.txt", embed_input, fmt, "MHA 0: ");
  MultiLayerPerceptron(embed_input, params_path.mlp_0);
  MatWriteFile("mlp_0_output.txt", embed_input, fmt, "MLP 0: ");

  MultiHeadAttention(embed_input, params_path.mha_1);
  MatWriteFile("mha_1_output.txt", embed_input, fmt, "MHA 1: ");
  MultiLayerPerceptron(embed_input, params_path.mlp_1);
  MatWriteFile("mlp_1_output.txt", embed_input, fmt, "MLP 1: ");

  Tensor output = MatDim(1, 5);
  Classifier(embed_input, output, params_path.classifier);
  MatUnDim(embed_input);
  MatWriteFile("output.txt", output, fmt, "Output: ");

  return 0;
}
