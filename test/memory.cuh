#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

#define CUDA_CALL(call)                                                                                      \
  do {                                                                                                       \
    auto err = call;                                                                                         \
    if (err != cudaSuccess) {                                                                                \
      std::cerr << "Cuda error in file " << __FILE__ << " L:" << __LINE__                                    \
                << "; Error: " << cudaGetErrorString(err) << '\n';                                           \
      throw std::runtime_error(cudaGetErrorString(err));                                                     \
    }                                                                                                        \
  } while (false)

template<class MemType>
class Memory
{
private:
  struct cuda_deleter
  {
    void operator()(MemType* ptr) const
    {
      if (ptr != nullptr) {
        cudaFree(ptr);
      }
    }
  };
  static std::unique_ptr<MemType, std::function<void(MemType*)>> allocate_cuda(
    const std::vector<MemType>& h_vec)
  {
    MemType* d_new_mem = nullptr;
    CUDA_CALL(cudaMalloc(&d_new_mem, h_vec.size() * sizeof(MemType)));
    return { d_new_mem, cuda_deleter{} };
  }

public:
  Memory(std::vector<MemType> init)
    : h_vec_{ std::move(init) }
    , d_mem_(allocate_cuda(h_vec_))
  {
    this->to_device();
  }
  Memory(std::size_t count, MemType init_val)
    : h_vec_(count, init_val)
    , d_mem_(allocate_cuda(h_vec_))
  {
    this->to_device();
  }

  ~Memory() = default;

  void to_device()
  {
    CUDA_CALL(
      cudaMemcpy(this->get_device(), this->get_host_const(), this->get_size_bytes(), cudaMemcpyHostToDevice));
  }

  void to_host()
  {
    CUDA_CALL(
      cudaMemcpy(this->get_host(), this->get_device_const(), this->get_size_bytes(), cudaMemcpyDeviceToHost));
  }

  MemType* get_host() { return h_vec_.data(); }

  MemType* get_device() { return d_mem_.get(); }

  const MemType* get_host_const() const { return h_vec_.data(); }

  const MemType* get_device_const() const { return d_mem_.get(); }

  std::size_t get_num_elems() const { return h_vec_.size(); }

  std::size_t get_size_bytes() const { return h_vec_.size() * sizeof(MemType); }

  std::vector<MemType> get_host_copy() const { return h_vec_; }

  const std::vector<MemType> get_host_vector() const { return h_vec_; }

  void set_memory_to(const std::vector<MemType>& other)
  {
    if (other.size() != h_vec_.size()) {
      throw std::length_error("Mismatching sizes!");
    }
    for (std::size_t i = 0; i < h_vec_.size(); ++i) {
      h_vec_[i] = other[i];
    }
    this->to_device();
  }

  bool is_device_matching_host() const
  {
    bool matching = true;
    std::vector<MemType> h_device_memory(this->get_num_elems());
    CUDA_CALL(cudaMemcpy(
      h_device_memory.data(), this->get_device_const(), this->get_size_bytes(), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < this->get_num_elems(); ++i) {
      if (h_device_memory[i] != h_vec_[i]) {
        matching = false;
      }
    }
    return matching;
  }

private:
  std::vector<MemType> h_vec_;
  std::unique_ptr<MemType, std::function<void(MemType*)>> d_mem_;
};

template<class BinOperator, class Arg0>
auto
fold(BinOperator&&, Arg0&& a0)
{
  return std::forward<Arg0>(a0);
}

template<class BinOperator, class Arg0, class... Args>
auto
fold(BinOperator&& bin_op, Arg0&& a0, Args&&... args)
{
  return bin_op(a0, fold(std::forward<Args>(args)...));
}

template<class... MemTypes>
bool
compare_gpu_cpu(const Memory<MemTypes>&... args)
{
  CUDA_CALL(cudaDeviceSynchronize());
  return fold(std::logical_and<>{}, true, args.is_device_matching_host()...);
}

