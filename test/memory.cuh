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

namespace detail {

class bool_vector
{
  struct memory_deleter
  {
    void operator()(bool* ptr) const
    {
      if (ptr != nullptr) {
        free(ptr);
      }
    }
  };

public:
  bool_vector(std::size_t num_elems)
    : size_{ num_elems }
    , storage_{ new bool[size_] }
  {
  }
  bool_vector(std::vector<bool> init)
    : bool_vector(init.size())
  {
    for (std::size_t i = 0; i < size_; ++i) {
      storage_[i] = init[i];
    }
  }
  bool_vector(std::size_t count, bool init_val)
    : bool_vector(count)
  {
    for (std::size_t i = 0; i < size_; ++i) {
      storage_[i] = init_val;
    }
  }

  bool_vector(const bool_vector&) = delete;
  bool_vector(bool_vector&& other)
    : size_{ other.size_ }
    , storage_{ other.storage_ }
  {
    other.size_ = 0;
    other.storage_ = nullptr;
  }
  bool_vector& operator=(const bool_vector&) = delete;
  bool_vector& operator=(bool_vector&& other)
  {
    delete[] storage_;
    this->size_ = other.size_;
    this->storage_ = other.storage_;
    other.size_ = 0;
    other.storage_ = nullptr;
    return *this;
  }

  ~bool_vector()
  {
    delete[] storage_;
    storage_ = nullptr;
  }

  std::vector<bool> to_std_vector() const
  {
    std::vector<bool> ret(size_);
    for (std::size_t i = 0; i < size_; ++i) {
      ret[i] = storage_[i];
    }
    return ret;
  }
  operator std::vector<bool>() const { return to_std_vector(); }

  bool& operator[](std::size_t idx) { return storage_[idx]; }
  const bool& operator[](std::size_t idx) const { return storage_[idx]; }

  bool& at(std::size_t idx) { return storage_[idx]; }
  const bool& at(std::size_t idx) const { return storage_[idx]; }

  std::size_t size() const { return size_; }
  bool* data() { return storage_; }
  const bool* data() const { return storage_; }
  const bool* const_data() const { return storage_; }

  bool* begin() { return storage_; }
  bool* end() { return storage_ + size_; }
  const bool* begin() const { return storage_; }
  const bool* end() const { return storage_ + size_; }
  const bool* cbegin() const { return storage_; }
  const bool* cend() const { return storage_ + size_; }

private:
  std::size_t size_;
  bool* storage_;
};

} // namespace detail

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

  using host_storage_vector =
    std::conditional_t<std::is_same<MemType, bool>::value, detail::bool_vector, std::vector<MemType>>;

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

  std::vector<MemType> get_device_copy() const
  {
    host_storage_vector h_device_memory(this->get_num_elems());
    CUDA_CALL(cudaMemcpy(
      h_device_memory.data(), this->get_device_const(), this->get_size_bytes(), cudaMemcpyDeviceToHost));
    return h_device_memory;
  }

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
    host_storage_vector h_device_memory = this->get_device_copy();
    for (std::size_t i = 0; i < this->get_num_elems(); ++i) {
      if (h_device_memory[i] != h_vec_[i]) {
        matching = false;
        std::cerr << i << ": host " << h_vec_[i] << " vs " << h_device_memory[i] << " device\n";
      }
    }
    return matching;
  }

  void print_device_host() const
  {
    host_storage_vector h_device_memory(this->get_num_elems());
    CUDA_CALL(cudaMemcpy(
      h_device_memory.data(), this->get_device_const(), this->get_size_bytes(), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < this->get_num_elems(); ++i) {
      std::cout << i << ": host " << h_vec_[i] << " vs " << h_device_memory[i] << " device\n";
    }
  }

private:
  host_storage_vector h_vec_;
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
  return bin_op(std::forward<Arg0>(a0), fold(bin_op, std::forward<Args>(args)...));
}

template<class... MemTypes>
bool
compare_gpu_cpu(const Memory<MemTypes>&... args)
{
  CUDA_CALL(cudaDeviceSynchronize());
  return fold(std::logical_and<bool>{}, true, args.is_device_matching_host()...);
}

