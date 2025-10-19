/*
    Tiny CNN primitives.
*/

#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <random>
#include <cmath>
#include <fstream>
#include <cstdint>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <omp.h>

// Global hyperparameters & model sizes
// Filters, kernel, output classes
const int F = 8;
const int K = 3;
const int O = 10;
// Training loop settings
const int trainN = 256; // number of samples to train on
const int batch = 32;   // batch size
const int epochs = 50;  // epochs
const float lr = 0.01f; // learning rate

/**
 * @brief Computes the flattened 1D index for a 4D tensor (N, C, H, W) in row-major order.
 *
 * @param n Index for the batch dimension (N).
 * @param c Index for the channel dimension (C).
 * @param h Index for the height dimension (H).
 * @param w Index for the width dimension (W).
 * @param C Total number of channels.
 * @param H Total height.
 * @param W Total width.
 * @return size_t The corresponding 1D index in the flattened array.
 */
static inline size_t idx4(int n, int c, int h, int w, int C, int H, int W)
{
    return ((size_t)n * C * H * W) + ((size_t)c * H * W) + ((size_t)h * W) + (size_t)w;
}

/**
 * @brief Performs a 2D convolution on input tensor in NCHW format with kernel size 3x3, padding=1, stride=1.
 *
 * This function assumes the kernel size K=3, padding=1, and stride=1, so the output height and width
 * are the same as the input. The input tensor is expected in NCHW layout (batch, channels, height, width).
 * The weights are expected in F x C x K x K layout, and the bias is per output channel.
 *
 * @param x Input tensor in NCHW format, flattened to a 1D vector.
 * @param N Batch size.
 * @param C Number of input channels.
 * @param H Input height.
 * @param W Input width.
 * @param w Convolution weights, flattened to a 1D vector (F x C x K x K).
 * @param b Bias vector, length F (can be empty for no bias).
 * @param F Number of output channels (filters).
 * @param K Kernel size (must be 3).
 * @return Output tensor in NCHW format, flattened to a 1D vector.
 */
std::vector<float> conv2d_nchw_pad1(
    const std::vector<float> &x, int N, int C, int H, int W,
    const std::vector<float> &w, const std::vector<float> &b, int F, int K)
{
    assert(K == 3 && "This minimal impl assumes K=3 for pad=1, stride=1");
    const int pad = 1;
    const int OH = H; // with pad=1 and stride=1, output H=W unchanged
    const int OW = W;
    std::vector<float> y((size_t)N * F * OH * OW, 0.0f);

#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; ++n)
    {
        for (int f = 0; f < F; ++f)
        {
            for (int i = 0; i < OH; ++i)
            {
                for (int j = 0; j < OW; ++j)
                {
                    float acc = b.empty() ? 0.0f : b[f];
                    for (int c = 0; c < C; ++c)
                    {
                        for (int ki = 0; ki < K; ++ki)
                        {
                            for (int kj = 0; kj < K; ++kj)
                            {
                                int in_i = i + ki - pad;
                                int in_j = j + kj - pad;
                                float xv = 0.0f;
                                if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W)
                                {
                                    xv = x[idx4(n, c, in_i, in_j, C, H, W)];
                                }
                                size_t widx = ((size_t)f * C * K * K) + ((size_t)c * K * K) + ((size_t)ki * K) + (size_t)kj;
                                float ww = w[widx];
                                acc += xv * ww;
                            }
                        }
                    }
                    y[idx4(n, f, i, j, F, OH, OW)] = acc;
                }
            }
        }
    }
    return y;
}

/**
 * @brief Applies the ReLU activation function in-place to a vector of floats.
 *
 * This function iterates over the input vector and sets any negative values to zero.
 * It is intended for use with data in NCHW format, but operates generically on the input vector.
 *
 * @param x Reference to a std::vector<float> containing the input data to be modified in-place.
 */
void relu_inplace_nchw(std::vector<float> &x)
{
#pragma omp parallel for
    for (auto &v : x)
        if (v < 0.0f)
            v = 0.0f;
}

/**
 * @brief Performs 2D max pooling with a 2x2 kernel and stride 2 on a 4D input tensor.
 *
 * This function applies max pooling to the input tensor `x` of shape (N, C, H, W),
 * where N is the batch size, C is the number of channels, H is the height, and W is the width.
 * The pooling operation uses a 2x2 window with a stride of 2, reducing the spatial dimensions
 * by half (OH = H / 2, OW = W / 2). The output tensor `y` has shape (N, C, OH, OW).
 *
 * @param x   Input tensor as a flattened std::vector<float> of shape (N * C * H * W).
 * @param N   Batch size.
 * @param C   Number of channels.
 * @param H   Input height (must be even).
 * @param W   Input width (must be even).
 * @param OH  Output height (set by the function to H / 2).
 * @param OW  Output width (set by the function to W / 2).
 * @return    Output tensor as a flattened std::vector<float> of shape (N * C * OH * OW).
 *
 * @note The function asserts that H and W are even numbers.
 * @note The helper function idx4 is assumed to compute the flat index for 4D tensors.
 */
std::vector<float> maxpool2d_2x2_s2(
    const std::vector<float> &x, int N, int C, int H, int W,
    int &OH, int &OW)
{
    assert(H % 2 == 0 && W % 2 == 0);
    OH = H / 2;
    OW = W / 2;
    std::vector<float> y((size_t)N * C * OH * OW, 0.0f);

#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int i = 0; i < OH; ++i)
            {
                for (int j = 0; j < OW; ++j)
                {
                    float m = -1e30f;
                    for (int di = 0; di < 2; ++di)
                    {
                        for (int dj = 0; dj < 2; ++dj)
                        {
                            int in_i = 2 * i + di;
                            int in_j = 2 * j + dj;
                            float v = x[idx4(n, c, in_i, in_j, C, H, W)];
                            if (v > m)
                                m = v;
                        }
                    }
                    y[idx4(n, c, i, j, C, OH, OW)] = m;
                }
            }
        }
    }
    return y;
}

/**
 * @brief Flattens a 4D tensor in NCHW format into a 1D vector.
 *
 * This function takes a tensor represented as a std::vector<float> with shape (N, C, H, W)
 * and returns a flattened 1D vector. The input data is assumed to be stored in contiguous
 * NCHW order. The shape information (N, C, H, W) is tracked externally.
 *
 * @param x The input tensor as a 1D vector of floats in NCHW order.
 * @param N The batch size (number of images).
 * @param C The number of channels.
 * @param H The height of each image.
 * @param W The width of each image.
 * @return A flattened 1D vector containing all elements of the input tensor.
 */
std::vector<float> flatten_nchw(const std::vector<float> &x, int N, int C, int H, int W)
{
    return x; // contiguous in this layout per image, but for clarity we keep as vector and track shape externally
}

/**
 * @brief Performs a linear (fully connected) forward pass.
 *
 * Computes the output Y = X * W + b for a batch of input vectors.
 *
 * @param X Input data as a flattened vector of size N * D, where N is the batch size and D is the input dimension.
 * @param N Number of input samples (batch size).
 * @param D Input dimension (number of features per sample).
 * @param W Weight matrix as a flattened vector of size D * O, where O is the output dimension.
 * @param b Bias vector of size O. If empty, no bias is added.
 * @param O Output dimension (number of output features per sample).
 * @return std::vector<float> Output data as a flattened vector of size N * O.
 */
std::vector<float> linear_forward(const std::vector<float> &X, int N, int D,
                                  const std::vector<float> &W, const std::vector<float> &b, int O)
{
    std::vector<float> Y((size_t)N * O, 0.0f);
#pragma omp parallel for
    for (int n = 0; n < N; ++n)
    {
        for (int o = 0; o < O; ++o)
        {
            float acc = b.empty() ? 0.0f : b[o];
            for (int d = 0; d < D; ++d)
            {
                acc += X[(size_t)n * D + d] * W[(size_t)d * O + o];
            }
            Y[(size_t)n * O + o] = acc;
        }
    }
    return Y;
}

/**
 * @brief Computes the softmax probabilities for a batch of logits.
 *
 * Given a flat vector of logits representing N samples, each with C classes,
 * this function applies the softmax operation to each sample independently.
 * The output is a vector of probabilities of the same shape as the input.
 *
 * @param logits A flat vector of input logits of size N * C.
 * @param N The number of samples (batch size).
 * @param C The number of classes per sample.
 * @return std::vector<float> A flat vector of softmax probabilities of size N * C.
 */
std::vector<float> softmax(const std::vector<float> &logits, int N, int C)
{
    std::vector<float> probs((size_t)N * C, 0.0f);
    for (int n = 0; n < N; ++n)
    {
        float m = -1e30f;
        for (int c = 0; c < C; ++c)
            m = std::max(m, logits[(size_t)n * C + c]);
        float sum = 0.0f;
        for (int c = 0; c < C; ++c)
        {
            float e = std::exp(logits[(size_t)n * C + c] - m);
            probs[(size_t)n * C + c] = e;
            sum += e;
        }
        float inv = 1.0f / sum;
        for (int c = 0; c < C; ++c)
            probs[(size_t)n * C + c] *= inv;
    }
    return probs;
}

/**
 * @brief Reads a 32-bit unsigned integer from a binary stream in big-endian order.
 *
 * This function reads 4 bytes from the provided input file stream and interprets them
 * as a big-endian 32-bit unsigned integer.
 *
 * @param ifs Reference to an input file stream to read from.
 * @return The 32-bit unsigned integer read from the stream in big-endian order.
 */
static uint32_t read_be_u32(std::ifstream &ifs)
{
    uint8_t b[4];
    ifs.read(reinterpret_cast<char *>(b), 4);
    return (uint32_t)b[0] << 24 | (uint32_t)b[1] << 16 | (uint32_t)b[2] << 8 | (uint32_t)b[3];
}

/**
 * @brief Loads MNIST image data from a binary file.
 *
 * This function reads MNIST image data from the specified file path and stores the normalized pixel values
 * (in the range [0.0, 1.0]) into the provided output vector. It also sets the number of images (N),
 * image height (H), and image width (W). The function supports loading up to a maximum number of images (maxN).
 *
 * @param path The path to the MNIST image file.
 * @param out Reference to a vector where the loaded and normalized image data will be stored.
 * @param N Reference to an integer that will be set to the number of images loaded.
 * @param H Reference to an integer that will be set to the image height (should be 28 for MNIST).
 * @param W Reference to an integer that will be set to the image width (should be 28 for MNIST).
 * @param maxN The maximum number of images to load (default is 1).
 * @return true if the images were loaded successfully; false otherwise.
 */
bool load_mnist_images(const std::string &path, std::vector<float> &out, int &N, int &H, int &W, int maxN = 1)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return false;
    uint32_t magic = read_be_u32(ifs);
    if (magic != 0x00000803)
        return false; // images
    uint32_t n = read_be_u32(ifs);
    uint32_t h = read_be_u32(ifs);
    uint32_t w = read_be_u32(ifs);
    if (h != 28 || w != 28)
        return false;
    N = std::min<int>(n, maxN);
    H = (int)h;
    W = (int)w;
    out.resize((size_t)N * H * W);

    for (int i = 0; i < N; ++i)
    {
        for (int p = 0; p < H * W; ++p)
        {
            unsigned char pix;
            ifs.read(reinterpret_cast<char *>(&pix), 1);
            out[(size_t)i * H * W + p] = (float)pix / 255.0f;
        }
    }
    return true;
}

/**
 * @brief Loads MNIST label data from a binary file.
 *
 * This function reads the MNIST label file specified by `path`, verifies its magic number,
 * and loads up to `maxN` labels into the provided `labels` vector. The actual number of labels
 * loaded is stored in `N`.
 *
 * @param path The path to the MNIST label file.
 * @param labels Reference to a vector where the loaded labels will be stored.
 * @param N Reference to an integer where the number of labels loaded will be set.
 * @param maxN Maximum number of labels to load (default is 1).
 * @return true if the labels were loaded successfully; false otherwise.
 */
bool load_mnist_labels(const std::string &path, std::vector<uint8_t> &labels, int &N, int maxN = 1)
{
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs)
        return false;
    uint32_t magic = read_be_u32(ifs);
    if (magic != 0x00000801)
        return false; // labels
    uint32_t n = read_be_u32(ifs);
    N = std::min<int>(n, maxN);
    labels.resize(N);
    ifs.read(reinterpret_cast<char *>(labels.data()), N);
    return (bool)ifs;
}

/**
 * @brief Computes the cross-entropy loss and its gradient with respect to logits.
 *
 * This function calculates the average cross-entropy loss between predicted logits and integer class labels
 * for a batch of samples. It also computes the gradient of the loss with respect to the logits (dlogits).
 * The softmax operation is performed in a numerically stable way for each sample.
 *
 * @param logits   A vector of floats containing the raw prediction scores (logits) for each class and sample.
 *                 The size should be N * C, where N is the batch size and C is the number of classes.
 * @param labels   A vector of integers containing the ground truth class indices for each sample. Size is N.
 * @param N        The number of samples in the batch.
 * @param C        The number of classes.
 * @param dlogits  Output vector to store the gradient of the loss with respect to the logits.
 *                 Will be resized to N * C and filled with the computed gradients.
 * @return         The average cross-entropy loss over the batch.
 */
float cross_entropy_with_logits_grad(const std::vector<float> &logits, const std::vector<int> &labels,
                                     int N, int C, std::vector<float> &dlogits)
{
    dlogits.assign((size_t)N * C, 0.0f);
    float loss = 0.0f;
    for (int n = 0; n < N; ++n)
    {
        // compute stable softmax for this sample and accumulate gradient
        float m = -1e30f;
        for (int c = 0; c < C; ++c)
            m = std::max(m, logits[(size_t)n * C + c]);
        float sum = 0.0f;
        for (int c = 0; c < C; ++c)
            sum += std::exp(logits[(size_t)n * C + c] - m);
        float log_sum = std::log(sum);
        int y = labels[n];
        float logpy = logits[(size_t)n * C + y] - m - log_sum;
        loss += -logpy;
        // grad: softmax - onehot
        for (int c = 0; c < C; ++c)
        {
            float p = std::exp(logits[(size_t)n * C + c] - m - log_sum);
            dlogits[(size_t)n * C + c] = p;
        }
        dlogits[(size_t)n * C + y] -= 1.0f;
    }
    // average over batch
    float invN = 1.0f / (float)N;

#pragma omp parallel for
    for (auto &v : dlogits)
        v *= invN;
    return loss * invN;
}

/**
 * @brief Computes the backward pass for a linear (fully connected) layer.
 *
 * Given the input X, weights W, and upstream gradients dY, this function calculates
 * the gradients with respect to the input (dX), weights (dW), and biases (db).
 *
 * @param X Input data matrix of shape (N, D), flattened as a vector.
 * @param N Number of samples in the batch.
 * @param D Number of input features.
 * @param W Weight matrix of shape (D, O), flattened as a vector.
 * @param O Number of output features.
 * @param dY Upstream gradients of shape (N, O), flattened as a vector.
 * @param dX Output vector to store gradients with respect to input X (shape: N * D).
 * @param dW Output vector to store gradients with respect to weights W (shape: D * O).
 * @param db Output vector to store gradients with respect to biases (shape: O).
 */
void linear_backward(const std::vector<float> &X, int N, int D,
                     const std::vector<float> &W, int O,
                     const std::vector<float> &dY,
                     std::vector<float> &dX, std::vector<float> &dW, std::vector<float> &db)
{
    dX.assign((size_t)N * D, 0.0f);
    dW.assign((size_t)D * O, 0.0f);
    db.assign((size_t)O, 0.0f);

#pragma omp parallel
    {
        // acumuladores locais por thread
        std::vector<float> dW_local((size_t)D * O, 0.0f);
        std::vector<float> db_local((size_t)O, 0.0f);

// cada thread pega um subconjunto de n; dX[n,*] é exclusivo e pode escrever direto
#pragma omp for schedule(static)
        for (int n = 0; n < N; ++n)
        {
            for (int o = 0; o < O; ++o)
            {
                float g = dY[(size_t)n * O + o];
                db_local[o] += g;
                for (int d = 0; d < D; ++d)
                {
                    dW_local[(size_t)d * O + o] += X[(size_t)n * D + d] * g;
                    dX[(size_t)n * D + d] += W[(size_t)d * O + o] * g;
                }
            }
        }

// fusão segura
#pragma omp critical
        {
            for (size_t i = 0; i < dW.size(); ++i)
                dW[i] += dW_local[i];
            for (int o = 0; o < O; ++o)
                db[o] += db_local[o];
        }
    }
}

/**
 * @brief Backward pass for ReLU activation layer.
 */
void relu_backward_inplace(std::vector<float> &dY, const std::vector<float> &Y)
{
    const size_t sz = dY.size();

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < sz; ++i)
        if (Y[i] <= 0.0f)
            dY[i] = 0.0f;
}

/**
 * @brief Backward pass for 2D max pooling layer.
 *
 * This function computes the gradient of the input tensor for a 2D max pooling
 * layer during the backward pass. It uses the input tensor to find the positions
 * of the maximum values in each pooling region and propagates the gradients from
 * the output tensor to the input tensor.
 *
 * @param X Input tensor (NCHW format).
 * @param N Batch size.
 * @param C Number of channels.
 * @param H Height of the input.
 * @param W Width of the input.
 * @param dY Gradient of the output tensor (NCHW format).
 * @param OH Height of the output.
 * @param OW Width of the output.
 * @return Gradient of the input tensor (NCHW format).
 */
std::vector<float> maxpool2d_2x2_s2_backward(
    const std::vector<float> &X, int N, int C, int H, int W,
    const std::vector<float> &dY, int OH, int OW)
{
    std::vector<float> dX((size_t)N * C * H * W, 0.0f);

#pragma omp parallel for collapse(4) schedule(static)
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int i = 0; i < OH; ++i)
            {
                for (int j = 0; j < OW; ++j)
                {
                    // find argmax in 2x2 window
                    int bi = 2 * i, bj = 2 * j;
                    float m = -1e30f;
                    int mi = bi, mj = bj;
                    for (int di = 0; di < 2; ++di)
                    {
                        for (int dj = 0; dj < 2; ++dj)
                        {
                            int ii = bi + di, jj = bj + dj;
                            float v = X[idx4(n, c, ii, jj, C, H, W)];
                            if (v > m)
                            {
                                m = v;
                                mi = ii;
                                mj = jj;
                            }
                        }
                    }
                    dX[idx4(n, c, mi, mj, C, H, W)] += dY[idx4(n, c, i, j, C, OH, OW)];
                }
            }
        }
    }
    return dX;
}

/**
 * @brief Backward pass for 2D convolution with NCHW format and padding.
 *
 * This function computes the gradients of the input, weights, and biases for a
 * 2D convolution layer during the backward pass. It takes into account the
 * padding and the dimensions of the input and output tensors.
 *
 * @param X Input tensor (NCHW format).
 * @param N Batch size.
 * @param C Number of input channels.
 * @param H Height of the input.
 * @param W Width of the input.
 * @param Wc Convolutional weights tensor (FCKK format).
 * @param dWc Gradient of the convolutional weights.
 * @param dbc Gradient of the convolutional biases.
 * @param F Number of output filters.
 * @param K Kernel size (assumed to be square).
 * @param dY Gradient of the output tensor (NCHW format).
 * @param dX Gradient of the input tensor (NCHW format).
 */
void conv2d_nchw_pad1_backward(
    const std::vector<float> &X, int N, int C, int H, int W,
    const std::vector<float> &Wc, std::vector<float> &dWc, std::vector<float> &dbc, int F, int K,
    const std::vector<float> &dY,
    std::vector<float> &dX)
{
    const int pad = 1;
    const int OH = H, OW = W;

    dX.assign((size_t)N * C * H * W, 0.0f);
    dWc.assign((size_t)F * C * K * K, 0.0f);
    dbc.assign((size_t)F, 0.0f);

#pragma omp parallel
    {
        std::vector<float> dWc_local((size_t)F * C * K * K, 0.0f);
        std::vector<float> dbc_local((size_t)F, 0.0f);

// distribui (n,f,i); itera j por dentro
#pragma omp for collapse(3) schedule(static)
        for (int n = 0; n < N; ++n)
        {
            for (int f = 0; f < F; ++f)
            {
                for (int i = 0; i < OH; ++i)
                {
                    for (int j = 0; j < OW; ++j)
                    {
                        float g = dY[idx4(n, f, i, j, F, OH, OW)];
                        dbc_local[f] += g;

                        for (int c = 0; c < C; ++c)
                        {
                            for (int ki = 0; ki < K; ++ki)
                            {
                                for (int kj = 0; kj < K; ++kj)
                                {
                                    int in_i = i + ki - pad;
                                    int in_j = j + kj - pad;
                                    if (in_i >= 0 && in_i < H && in_j >= 0 && in_j < W)
                                    {
                                        size_t xidx = idx4(n, c, in_i, in_j, C, H, W);
                                        size_t widx = ((size_t)f * C * K * K) + ((size_t)c * K * K) + ((size_t)ki * K) + (size_t)kj;
                                        float xv = X[xidx];

                                        dWc_local[widx] += xv * g;

// dX recebe contribuições de múltiplos (f,i,j) -> atomic
#pragma omp atomic
                                        dX[xidx] += Wc[widx] * g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

// fusão
#pragma omp critical
        {
            for (size_t i = 0; i < dWc.size(); ++i)
                dWc[i] += dWc_local[i];
            for (int f = 0; f < F; ++f)
                dbc[f] += dbc_local[f];
        }
    }
}

/**
 * @brief Saves a model to a text file.
 *
 * The function writes the model parameters and weights to a text file specified by `path`.
 * It includes the dimensions (F, C, K, O, D) followed by the weights and biases for the
 * convolutional and linear layers. Each value is written with a precision of 8 decimal places.
 *
 * @param path Path to save the model text file.
 * @param Wc Convolutional weights vector.
 * @param bc Convolutional biases vector.
 * @param Wl Linear weights vector.
 * @param bl Linear biases vector.
 * @param F Number of convolutional filters.
 * @param C Number of input channels.
 * @param K Kernel size for convolution.
 * @param O Number of output classes.
 * @param D Number of inputs to the linear layer.
 * @return true if the model was saved successfully; false otherwise.
 */
bool save_model_text(const std::string &path,
                     const std::vector<float> &Wc, const std::vector<float> &bc,
                     const std::vector<float> &Wl, const std::vector<float> &bl,
                     int F, int C, int K, int O, int D)
{
    std::ofstream ofs(path);
    if (!ofs)
        return false;
    ofs << F << ' ' << C << ' ' << K << ' ' << O << ' ' << D << '\n';
    for (auto v : Wc)
        ofs << std::setprecision(8) << v << ' ';
    ofs << '\n';
    for (auto v : bc)
        ofs << std::setprecision(8) << v << ' ';
    ofs << '\n';
    for (auto v : Wl)
        ofs << std::setprecision(8) << v << ' ';
    ofs << '\n';
    for (auto v : bl)
        ofs << std::setprecision(8) << v << ' ';
    ofs << '\n';
    return true;
}

/**
 * @brief Loads a model from a text file and populates the provided weight and bias vectors.
 *
 * The function reads model parameters and weights from a text file specified by `path`.
 * It expects the file to begin with five integers representing the dimensions (F, C, K, O, D),
 * which must match the expected values (`Fexp`, `Cexp`, `Kexp`, `Oexp`, `Dexp`). If the dimensions
 * do not match or the file cannot be opened/read, the function returns false.
 *
 * The function then reads the convolutional weights (`Wc`), convolutional biases (`bc`),
 * linear weights (`Wl`), and linear biases (`bl`) from the file, resizing the vectors as needed.
 *
 * @param path Path to the model text file.
 * @param Wc Reference to vector to store convolutional weights.
 * @param bc Reference to vector to store convolutional biases.
 * @param Wl Reference to vector to store linear weights.
 * @param bl Reference to vector to store linear biases.
 * @param Fexp Expected number of convolutional filters.
 * @param Cexp Expected number of input channels.
 * @param Kexp Expected kernel size.
 * @param Oexp Expected number of output classes.
 * @param Dexp Expected number of linear layer inputs.
 * @return true if the model was loaded successfully and dimensions match; false otherwise.
 */
bool load_model_text(const std::string &path,
                     std::vector<float> &Wc, std::vector<float> &bc,
                     std::vector<float> &Wl, std::vector<float> &bl,
                     int Fexp, int Cexp, int Kexp, int Oexp, int Dexp)
{
    std::ifstream ifs(path);
    if (!ifs)
        return false;
    int F, C, K, O, D;
    ifs >> F >> C >> K >> O >> D;
    if (!ifs || F != Fexp || C != Cexp || K != Kexp || O != Oexp || D != Dexp)
        return false;
    Wc.resize((size_t)F * C * K * K);
    bc.resize((size_t)F);
    Wl.resize((size_t)D * O);
    bl.resize((size_t)O);
    for (size_t i = 0; i < Wc.size(); ++i)
        ifs >> Wc[i];
    for (size_t i = 0; i < bc.size(); ++i)
        ifs >> bc[i];
    for (size_t i = 0; i < Wl.size(); ++i)
        ifs >> Wl[i];
    for (size_t i = 0; i < bl.size(); ++i)
        ifs >> bl[i];
    return (bool)ifs;
}

/**
 * @brief Evaluates a simple CNN model on the MNIST dataset.
 *
 * This function loads MNIST images and labels, processes them in batches,
 * performs a forward pass through a convolutional layer, ReLU activation,
 * max pooling, flattening, and a linear layer, then computes the classification
 * accuracy over the dataset.
 *
 * @param imgPath Path to the MNIST image file.
 * @param lblPath Path to the MNIST label file.
 * @param maxN Maximum number of samples to evaluate.
 * @param Wc Weights for the convolutional layer.
 * @param bc Biases for the convolutional layer.
 * @param Wl Weights for the linear layer.
 * @param bl Biases for the linear layer.
 * @param F Number of convolutional filters.
 * @param C Number of input channels.
 * @param H Height of input images.
 * @param W Width of input images.
 * @param K Kernel size for convolution.
 * @param O Number of output classes.
 * @return Classification accuracy as a percentage (0.0f to 100.0f), or -1.0f on error.
 */
float evaluate_on_mnist(const std::string &imgPath, const std::string &lblPath, int maxN,
                        const std::vector<float> &Wc, const std::vector<float> &bc,
                        const std::vector<float> &Wl, const std::vector<float> &bl,
                        int F, int C, int H, int W, int K, int O)
{
    std::vector<float> imgs;
    int N = 0, h = 0, w = 0;
    std::vector<uint8_t> lbls;
    int Nl = 0;
    if (!load_mnist_images(imgPath, imgs, N, h, w, maxN) || !load_mnist_labels(lblPath, lbls, Nl, maxN))
        return -1.0f;
    N = std::min(N, Nl);
    int correct = 0;
    const int H2 = H / 2, W2 = W / 2;
    const int D = F * H2 * W2;
    const int BATCH = 64;

    for (int s = 0; s < N; s += BATCH)
    {
        int B = std::min(BATCH, N - s);
        std::vector<float> Xb((size_t)B * C * H * W, 0.0f);

#pragma omp parallel for collapse(2) schedule(guided)
        for (int b = 0; b < B; ++b)
        {
            int idx = s + b;
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    Xb[idx4(b, 0, i, j, C, H, W)] = imgs[(size_t)idx * H * W + (size_t)i * W + j];
        }

        auto y1 = conv2d_nchw_pad1(Xb, B, C, H, W, Wc, bc, F, K);
        relu_inplace_nchw(y1);
        int H2o = 0, W2o = 0;
        auto y2 = maxpool2d_2x2_s2(y1, B, F, H, W, H2o, W2o);
        auto Xflat = flatten_nchw(y2, B, F, H2o, W2o);
        auto logits = linear_forward(Xflat, B, D, Wl, bl, O);

#pragma omp parallel for reduction(+ : correct)
        for (int b = 0; b < B; ++b)
        {
            int argm = 0;
            float best = logits[(size_t)b * O + 0];
            for (int c = 1; c < O; ++c)
            {
                float v = logits[(size_t)b * O + c];
                if (v > best)
                {
                    best = v;
                    argm = c;
                }
            }
            if (argm == (int)lbls[s + b])
                correct++;
        }
    }
    return 100.0f * correct / std::max(1, N);
}

int main()
{
    // Try to load one MNIST sample if file paths are given: ./cnn_minimal <images-idx3-ubyte> <labels-idx1-ubyte>
    int N = 1, C = 1, H = 28, W = 28;
    std::vector<float> x((size_t)N * C * H * W, 0.0f);
    int label = -1;
    bool used_mnist = false;

    // CLI args handling (minimal). We don't parse argc/argv here since Makefile's run doesn't forward args.
    // Instead, check environment variables MNIST_IMAGES and MNIST_LABELS; fallback to synthetic if not set or load fails.
    const char *env_imgs = std::getenv("MNIST_IMAGES");
    const char *env_lbls = std::getenv("MNIST_LABELS");
    if (env_imgs && env_lbls)
    {
        std::vector<float> imgs;
        std::vector<uint8_t> lbls;
        int nimg = 0, lh = 0, lw = 0, nlbl = 0;
        if (load_mnist_images(env_imgs, imgs, nimg, lh, lw, 1) && load_mnist_labels(env_lbls, lbls, nlbl, 1) && nimg >= 1 && nlbl >= 1)
        {
            // Move into x in NCHW
            for (int i = 0; i < H; ++i)
            {
                for (int j = 0; j < W; ++j)
                {
                    x[idx4(0, 0, i, j, C, H, W)] = imgs[(size_t)0 * H * W + (size_t)i * W + j];
                }
            }
            label = (int)lbls[0];
            used_mnist = true;
        }
    }

    if (!used_mnist)
    {
        // Synthetic input: a simple blob in the center
        for (int i = 10; i < 18; ++i)
        {
            for (int j = 10; j < 18; ++j)
            {
                x[idx4(0, 0, i, j, C, H, W)] = 1.0f;
            }
        }
    }

    // ----------------------- Minimal training loop (SGD) -----------------------
    // Uses file-scope constants for hyperparameters (F, K, O, trainN, batch, epochs, lr)

    // Load a small training set (fallback to default paths if env not set)
    std::string imgsPath = (std::getenv("MNIST_IMAGES") ? std::getenv("MNIST_IMAGES") : "mnist/train-images.idx3-ubyte");
    std::string lblsPath = (std::getenv("MNIST_LABELS") ? std::getenv("MNIST_LABELS") : "mnist/train-labels.idx1-ubyte");
    std::vector<float> imgsHW;
    int nimg = 0, h = 0, w = 0;
    std::vector<uint8_t> lblsU8;
    int nlbl = 0;
    bool loaded = load_mnist_images(imgsPath, imgsHW, nimg, h, w, trainN) && load_mnist_labels(lblsPath, lblsU8, nlbl, trainN);
    if (!loaded || nimg == 0 || nlbl == 0)
    {
        std::cout << "Training skipped (MNIST not available).\n";
        std::cout << "Done." << std::endl;
        return 0;
    }
    const int Ntrain = std::min(nimg, nlbl);
    std::vector<int> labelsInt(Ntrain);
    for (int i = 0; i < Ntrain; ++i)
        labelsInt[i] = (int)lblsU8[i];

    // Initialize weights (He init for ReLU)
    std::mt19937 rng(42);
    std::normal_distribution<float> ndc(0.0f, std::sqrt(2.0f / (C * 3.0f * 3.0f)));
    std::normal_distribution<float> ndl(0.0f, std::sqrt(2.0f / (F * 14.0f * 14.0f)));
    std::vector<float> Wc((size_t)F * C * K * K);
    for (auto &w_ : Wc)
        w_ = ndc(rng);
    std::vector<float> bc(F, 0.0f);
    const int D = F * 14 * 14; // after pool
    std::vector<float> Wl((size_t)D * O);
    for (auto &w_ : Wl)
        w_ = ndl(rng);
    std::vector<float> bl(O, 0.0f);

    // Indices for shuffling
    std::vector<int> indices(Ntrain);
    for (int i = 0; i < Ntrain; ++i)
        indices[i] = i;

    auto get_batch = [&](int start, int B, std::vector<float> &Xbatch, std::vector<int> &Ybatch)
    {
        Xbatch.assign((size_t)B * C * H * W, 0.0f);
        Ybatch.resize(B);
        for (int b = 0; b < B; ++b)
        {
            int idx = indices[start + b];
            Ybatch[b] = labelsInt[idx];
            // copy image idx (H*W) into (b,0,:,:)
            for (int i = 0; i < H; ++i)
            {
                for (int j = 0; j < W; ++j)
                {
                    Xbatch[idx4(b, 0, i, j, C, H, W)] = imgsHW[(size_t)idx * H * W + (size_t)i * W + j];
                }
            }
        }
    };

    // Training loop
    float epoch_loss = 0.0f;
    int epoch_count = 0;
    int correct = 0;
    int total = 0;
    for (int ep = 0; ep < epochs; ++ep)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        for (int s = 0; s < Ntrain; s += batch)
        {
            int B = std::min(batch, Ntrain - s);
            std::vector<float> Xb;
            std::vector<int> Yb;
            get_batch(s, B, Xb, Yb);

            // Forward
            auto y1 = conv2d_nchw_pad1(Xb, B, C, H, W, Wc, bc, F, K);
            relu_inplace_nchw(y1);
            int H2 = 0, W2 = 0;
            auto y2 = maxpool2d_2x2_s2(y1, B, F, H, W, H2, W2); // 14x14
            const int Dcur = F * H2 * W2;
            (void)Dcur;
            auto Xflat = flatten_nchw(y2, B, F, H2, W2);
            auto logits = linear_forward(Xflat, B, D, Wl, bl, O);

            // Loss + gradient
            std::vector<float> dlogits;
            float loss = cross_entropy_with_logits_grad(logits, Yb, B, O, dlogits);
            epoch_loss += loss;
            epoch_count += 1;

            // Accuracy (quick)
            for (int n = 0; n < B; ++n)
            {
                int argm = 0;
                float best = logits[(size_t)n * O + 0];
                for (int c = 1; c < O; ++c)
                {
                    float v = logits[(size_t)n * O + c];
                    if (v > best)
                    {
                        best = v;
                        argm = c;
                    }
                }
                correct += (argm == Yb[n]);
                total += 1;
            }

            // Backward
            std::vector<float> dXflat, dWl, dbl;
            linear_backward(Xflat, B, D, Wl, O, dlogits, dXflat, dWl, dbl);
            // dXflat -> dy2 (same buffer shape)
            auto dy2 = dXflat; // reinterpret as (B,F,H2,W2)
            auto dx1_after_pool = maxpool2d_2x2_s2_backward(y1, B, F, H, W, dy2, H2, W2);
            // ReLU backward
            relu_backward_inplace(dx1_after_pool, y1);
            // Conv backward
            std::vector<float> dWc, dbc, dX;
            conv2d_nchw_pad1_backward(Xb, B, C, H, W, Wc, dWc, dbc, F, K, dx1_after_pool, dX);

            // SGD step
            for (size_t i = 0; i < Wl.size(); ++i)
                Wl[i] -= lr * dWl[i];
            for (int i = 0; i < O; ++i)
                bl[i] -= lr * dbl[i];
            for (size_t i = 0; i < Wc.size(); ++i)
                Wc[i] -= lr * dWc[i];
            for (int i = 0; i < F; ++i)
                bc[i] -= lr * dbc[i];

            if (((s / batch) % 5) == 0)
            {
                std::cout << "Epoch " << ep + 1 << ", batch " << (s / batch) << ": loss=" << std::fixed << std::setprecision(4) << loss
                          << ", running acc=" << std::setprecision(2) << (100.0 * correct / std::max(1, total)) << "%\n";
            }
        }
    }

    std::cout << "Training done. Avg loss=" << std::fixed << std::setprecision(4) << (epoch_loss / std::max(1, epoch_count))
              << ", train acc=" << std::setprecision(2) << (100.0 * correct / std::max(1, total)) << "%\n";

    // Evaluate on test set (limit to 1000 samples for speed)
    float testAcc = evaluate_on_mnist("mnist/t10k-images.idx3-ubyte", "mnist/t10k-labels.idx1-ubyte", 1000,
                                      Wc, bc, Wl, bl, F, C, H, W, K, O);
    if (testAcc >= 0.0f)
    {
        std::cout << "Test accuracy (1000 samples): " << std::setprecision(2) << testAcc << "%\n";
    }
    else
    {
        std::cout << "Test set not found; skipping evaluation.\n";
    }

    // Save model
    std::string weightsOut = "bin/mnist_cnn_weights.txt";
    if (save_model_text(weightsOut, Wc, bc, Wl, bl, F, C, K, O, D))
    {
        std::cout << "Saved weights to " << weightsOut << "\n";
    }
    else
    {
        std::cout << "Failed to save weights to " << weightsOut << "\n";
    }

    // Quick inference on the very first training sample
    std::vector<float> Xtest((size_t)1 * C * H * W, 0.0f);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            Xtest[idx4(0, 0, i, j, C, H, W)] = imgsHW[(size_t)0 * H * W + (size_t)i * W + j];
    auto t1 = conv2d_nchw_pad1(Xtest, 1, C, H, W, Wc, bc, F, K);
    relu_inplace_nchw(t1);
    int H2t = 0, W2t = 0;
    auto t2 = maxpool2d_2x2_s2(t1, 1, F, H, W, H2t, W2t);
    auto Xft = flatten_nchw(t2, 1, F, H2t, W2t);
    auto logt = linear_forward(Xft, 1, D, Wl, bl, O);
    int pred = 0;
    float bestv = logt[0];
    for (int c = 1; c < O; ++c)
        if (logt[c] > bestv)
        {
            bestv = logt[c];
            pred = c;
        }
    std::cout << "First sample true=" << labelsInt[0] << ", pred=" << pred << "\n";

    // Demonstrate load model and predict again for consistency
    std::vector<float> Wc2, bc2, Wl2, bl2;
    if (load_model_text(weightsOut, Wc2, bc2, Wl2, bl2, F, C, K, O, D))
    {
        auto t1b = conv2d_nchw_pad1(Xtest, 1, C, H, W, Wc2, bc2, F, K);
        relu_inplace_nchw(t1b);
        int H2b = 0, W2b = 0;
        auto t2b = maxpool2d_2x2_s2(t1b, 1, F, H, W, H2b, W2b);
        auto Xfb = flatten_nchw(t2b, 1, F, H2b, W2b);
        auto logb = linear_forward(Xfb, 1, D, Wl2, bl2, O);
        int pred2 = 0;
        float best2 = logb[0];
        for (int c = 1; c < O; ++c)
            if (logb[c] > best2)
            {
                best2 = logb[c];
                pred2 = c;
            }
        std::cout << "Reloaded weights: first sample pred=" << pred2 << "\n";
    }

    std::cout << "Done." << std::endl;
    return 0;
}
