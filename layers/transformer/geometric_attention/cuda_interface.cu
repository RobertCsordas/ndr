#include <torch/extension.h>

__global__ void k_cuda_log_sigmoid_forward(int N, float * t, float *out_sigm, float *out_one_minus_sigm){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<N){
        float x = t[i];
        float c = - log(exp(-abs(x)) + 1);
        out_sigm[i] = min(x, 0.0f) + c;
        out_one_minus_sigm[i] = -max(x, 0.0f) + c;
    }
}

__global__ void k_cuda_log_sigmoid_backward(int N, float *t, float *grad_sigm, float *grad_one_minus_sigm, float *grad_out){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i<N){
        float x = t[i];
        float ne = exp(-abs(x));
        float coeff = 1.0 / (ne + 1.0) * ne;

        float r_one_minus = (x > 0) ? (coeff - 1) : (-coeff);
        float r =  (x < 0) ? (coeff - 1) : (-coeff);
        grad_out[i] = - grad_sigm[i] * r + grad_one_minus_sigm[i] * r_one_minus;
    }
}

std::vector<torch::Tensor> cuda_log_sigmoid_forward(torch::Tensor input){
    auto o1 = torch::empty_like(input);
    auto o2 = torch::empty_like(input);
    auto inf = input.flatten();

    const int N = inf.size(0);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    k_cuda_log_sigmoid_forward<<<blocks, threads>>>(N,  
        input.data<float>(),
        o1.data<float>(),
        o2.data<float>());
    
    return {o1, o2};
}

std::vector<torch::Tensor> cuda_log_sigmoid_backward(torch::Tensor input, torch::Tensor grad_sigm, torch::Tensor grad_one_minus_sigm){
    auto output = torch::empty_like(input);
    auto N = input.flatten().size(0);

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    k_cuda_log_sigmoid_backward<<<blocks, threads>>>(N,  
        input.data<float>(),
        grad_sigm.data<float>(),
        grad_one_minus_sigm.data<float>(),
        output.data<float>());
    
    return {output};
}


typedef torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> float_accessor;

__global__ void k_cuda_window_sum_forward(float_accessor csum, float_accessor out, int offset){
    const int in_p = threadIdx.z + blockIdx.z * blockDim.z;
    const int out_p_mem = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.x + blockIdx.x * blockDim.x;

    const int out_p = out_p_mem + offset;

    if (batch < out.size(0) & out_p_mem < out.size(1) & in_p < out.size(2)){
        float res;
        if (in_p == out_p){
            res = 0;
        } else {
            const int offset = abs(out_p - in_p);
            int p_i = out_p + offset - int(in_p > out_p);
            const int n_i = out_p - offset;

            p_i = min(p_i, out.size(2) - 1);

            float d_n = (n_i >= 0) ? (csum[batch][out_p_mem][n_i]) : 0.0;
            res = (csum[batch][out_p_mem][p_i]) - d_n;
        }

        out[batch][out_p_mem][in_p] = res;
    }

}

__global__ void k_cuda_window_sum_backward(float_accessor grad_in, float_accessor grad_out, int offset){
    const int in_p = threadIdx.z + blockIdx.z * blockDim.z;
    const int out_p_mem = threadIdx.y + blockIdx.y * blockDim.y;
    const int batch = threadIdx.x + blockIdx.x * blockDim.x;

    const int out_p = out_p_mem + offset;

    if (batch < grad_out.size(0) & out_p_mem < grad_out.size(1) & in_p < grad_out.size(2)){
        const int other = 2 * out_p - in_p;

        float res;
        if (in_p == grad_out.size(2) - 1){
            res = 0;
            for (int i = 0; i < other + int(in_p != out_p); ++i){
                res += grad_in[batch][out_p_mem][i];
            }
        } else if (in_p == out_p){
            res = grad_in[batch][out_p_mem][min(in_p + 1, grad_out.size(2) - 1)];
        } else if (in_p < out_p){
            res = -grad_in[batch][out_p_mem][in_p];
            if (other < grad_in.size(2))
                res -= grad_in[batch][out_p_mem][other];
        } else {
            res = grad_in[batch][out_p_mem][in_p + 1];
            if (other >= 0)
                res += grad_in[batch][out_p_mem][other];
        }

        grad_out[batch][out_p_mem][in_p] = res;
    }
}

dim3 get_grid_size(torch::Tensor target, dim3 block_dim){
    return dim3(
        (target.size(0) + block_dim.x - 1) / block_dim.x,
        (target.size(1) + block_dim.y - 1) / block_dim.y,
        (target.size(2) + block_dim.z - 1) / block_dim.z
    );
}

torch::Tensor cuda_window_sum_forward(torch::Tensor input, int offset){
    auto out = torch::empty_like(input);

    dim3 block_size(2, 2, 32);
    k_cuda_window_sum_forward<<<get_grid_size(input, block_size), block_size>>>(
        input.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        offset
    );
    
    return out;
}

torch::Tensor cuda_window_sum_backward(torch::Tensor grad_in, int offset){
    auto out = torch::empty_like(grad_in);

    dim3 block_size(2, 2, 32);
    k_cuda_window_sum_backward<<<get_grid_size(grad_in, block_size), block_size>>>(
        grad_in.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        out.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        offset
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cuda_log_sigmoid_forward",
        &cuda_log_sigmoid_forward,
        "Log sigmoid, forward pass"
    );
    m.def(
        "cuda_log_sigmoid_backward",
        &cuda_log_sigmoid_backward,
        "Log sigmoid, backward pass"
    );
    m.def(
        "cuda_window_sum_forward",
        &cuda_window_sum_forward,
        "Window sum, forward pass"
    );
    m.def(
        "cuda_window_sum_backward",
        &cuda_window_sum_backward,
        "Window sum, backward pass"
    );
}