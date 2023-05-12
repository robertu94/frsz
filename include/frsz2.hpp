#ifndef FRSZ_FRSZ2_CUH
#define FRSZ_FRSZ2_CUH

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#ifdef __CUDACC__
#define FRSZ_ATTRIBUTES __host__ __device__
#else
#define FRSZ_ATTRIBUTES
#endif
/* TODO
 * Remove dependency on __host__ constexpr functions (remove --expt-relaxed-constexpr compiler option)
 */

namespace frsz {

namespace detail {

// Only for device code
#ifdef __CUDA_ARCH__

template<class To, class From>
__device__ To
bit_cast_impl(const From& src) noexcept
{
  static_assert(std::is_trivially_constructible<To>::value, "Type To must be trivially constructable!");
  static_assert(sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                  std::is_trivially_copyable<To>::value,
                "Test");
  To dest;
  memcpy(&dest, &src, sizeof(From));
  return dest;
}

template<>
__device__ double
bit_cast_impl(const std::int64_t& src) noexcept
{
  return __longlong_as_double(src);
}

template<>
__device__ double
bit_cast_impl(const std::uint64_t& src) noexcept
{
  return __longlong_as_double(static_cast<std::int64_t>(src));
}

template<>
__device__ std::int64_t
bit_cast_impl(const double& src) noexcept
{
  return __double_as_longlong(src);
}

template<>
__device__ std::uint64_t
bit_cast_impl(const double& src) noexcept
{
  return static_cast<std::uint64_t>(__double_as_longlong(src));
}

template<>
__device__ float
bit_cast_impl(const std::int32_t& src) noexcept
{
  return __int_as_float(src);
}

template<>
__device__ float
bit_cast_impl(const std::uint32_t& src) noexcept
{
  return __uint_as_float(src);
}

template<>
__device__ std::int32_t
bit_cast_impl(const float& src) noexcept
{
  return __float_as_int(src);
}

template<>
__device__ std::uint32_t
bit_cast_impl(const float& src) noexcept
{
  return __float_as_uint(src);
}

#else // not device-code

template<class To, class From>
std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                   std::is_trivially_copyable<To>::value,
                 To>
bit_cast_impl(const From& src) noexcept
{
  static_assert(std::is_trivially_constructible<To>::value, "Type To must be trivially constructable!");
  To dest;
  std::memcpy(&dest, &src, sizeof(From));
  return dest;
}

#endif

} // namespace detail

namespace xstd {

template<class To, class From>
FRSZ_ATTRIBUTES std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable<From>::value &&
                                   std::is_trivially_copyable<To>::value,
                                 To>
bit_cast(const From& src) noexcept
{
  static_assert(std::is_trivially_constructible<To>::value, "Type To must be trivially constructable!");
  return detail::bit_cast_impl<To>(src);
}


namespace detail {

#if defined(__CUDA_ARCH__)
__device__ int countl_zero(std::uint64_t val) noexcept
{
  static_assert(sizeof(long long int) == sizeof(std::uint64_t), "Sizes must match!");
  return __clzll(val);
}
__device__ int countl_zero(std::uint32_t val) noexcept
{
  static_assert(sizeof(int) == sizeof(std::uint32_t), "Sizes must match!");
  return __clz(val);
}
__device__ int countl_zero(std::uint16_t val) noexcept
{
  return __clz(val) - 16;
}
__device__ int countl_zero(std::uint8_t val) noexcept
{
  return __clz(val) - 24;
}

#else  // !defined(__CUDA_ARCH__)

#if defined(__has_builtin)
#if __has_builtin(__builtin_clzll)
int countl_zero(unsigned long long val) noexcept
{
  return val == 0 ? CHAR_BIT * sizeof(unsigned long long) : __builtin_clzll(val);
}
#endif
#if __has_builtin(__builtin_clzl)
int countl_zero(unsigned long int val) noexcept
{
  return val == 0 ? CHAR_BIT * sizeof(unsigned int) : __builtin_clzl(val);
}
#endif
#if __has_builtin(__builtin_clz)
int countl_zero(unsigned int val) noexcept
{
  return val == 0 ? CHAR_BIT * sizeof(unsigned int) : __builtin_clz(val);
}
int countl_zero(std::uint16_t val) noexcept
{
  return __builtin_clz(val) - (CHAR_BIT * (sizeof(unsigned int) - sizeof(std::uint16_t)));
}
int countl_zero(std::uint8_t val) noexcept
{
  return __builtin_clz(val) - (CHAR_BIT * (sizeof(unsigned int) - sizeof(std::uint8_t)));
}
#endif
#else
#warning "no __has_builtin"
// default implementation
template<class T>
FRSZ_ATTRIBUTES std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, int>
countl_zero(T val) noexcept
{
  return val == 0 ? sizeof(T) * CHAR_BIT : countl_zero(val >> 1) - 1;
}
#endif // defined(__has_builtin)

#endif // defined(__CUDA_ARCH__)

} // namespace detail

// TODO call device function internally
template<class T>
FRSZ_ATTRIBUTES std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, int>
countl_zero(T val) noexcept
{
  return detail::countl_zero(val);
}

} // namespace xstd

namespace detail {

template<class T>
constexpr FRSZ_ATTRIBUTES std::enable_if_t<std::is_integral<T>::value, T>
get_next_power_of_two_value_impl(const T input, const T candidate = 1)
{
  return candidate >= input ? candidate : get_next_power_of_two_value_impl(input, candidate << 1);
}

template<class T>
constexpr FRSZ_ATTRIBUTES std::enable_if_t<std::is_integral<T>::value, T>
get_next_power_of_two_value(const T input)
{
  return get_next_power_of_two_value_impl(input);
}

template<class T>
using scaled_t =
  std::conditional_t<sizeof(T) == 8,
                     std::uint64_t,
                     std::conditional_t<sizeof(T) == 4,
                                        std::uint32_t,
                                        std::conditional_t<sizeof(T) == 2, std::uint16_t, std::uint32_t>>>;

template<size_t N>
using storage_t = std::conditional_t<
  (N <= 8),
  std::uint8_t,
  std::conditional_t<(N <= 16), std::uint16_t, std::conditional_t<(N <= 32), std::uint32_t, std::uint64_t>>>;

template<class T>
struct ones_s
{
  static_assert(std::is_integral<T>::value, "T must be an integer type!");
  static constexpr std::make_unsigned_t<T> value = ~std::make_unsigned_t<T>{};
};

namespace fp {

template<class T>
struct float_traits
{};
template<>
struct float_traits<float>
{
  using sign_t = bool;
  using exponent_t = std::int8_t;
  using significand_t = std::uint32_t;
  constexpr static std::int64_t sign_bits = 1;
  constexpr static std::int64_t exponent_bits = 8;
  constexpr static std::int64_t significand_bits = 23;
};
template<>
struct float_traits<double>
{
  using sign_t = bool;
  using exponent_t = std::int16_t;
  using significand_t = std::uint64_t;
  constexpr static std::int64_t sign_bits = 1;
  constexpr static std::int64_t exponent_bits = 11;
  constexpr static std::int64_t significand_bits = 52;
};

template<class T>
struct ebias_s
{
  constexpr static int value = ((1 << (float_traits<T>::exponent_bits - 1)) - 1);
};

// TODO check for intrinsic functions

template<class T>
FRSZ_ATTRIBUTES typename float_traits<T>::exponent_t
exponent(T f)
{
  constexpr std::size_t mantissa_shift = float_traits<T>::significand_bits + float_traits<T>::sign_bits;
  return (xstd::bit_cast<scaled_t<T>>(f) >> float_traits<T>::significand_bits &
          (ones_s<scaled_t<T>>::value >> mantissa_shift)) -
         ebias_s<T>::value;
}
template<class T>
FRSZ_ATTRIBUTES typename float_traits<T>::significand_t
significand(T f)
{
  scaled_t<T> s = xstd::bit_cast<scaled_t<T>>(f) &
                  (ones_s<scaled_t<T>>::value >> (sizeof(T) * CHAR_BIT - float_traits<T>::significand_bits));
  if (!(exponent(f) == -ebias_s<T>::value)) {
    // is not subnormal, add leading 1 bit
    s |= 1ull << float_traits<T>::significand_bits;
  }
  return s;
}
template<class T>
FRSZ_ATTRIBUTES typename float_traits<T>::sign_t
sign(T f)
{
  return (xstd::bit_cast<scaled_t<T>>(f) & (1ull << (CHAR_BIT * sizeof(T) - 1)));
}
constexpr bool positive = false;
constexpr bool negative = true;

template<class T>
std::int16_t
is_normal(T f)
{
  if (!(exponent(f) == -ebias_s<T>::value)) {
    return 1;
  }
  return 0;
}

template<class T>
FRSZ_ATTRIBUTES std::enable_if_t<std::is_floating_point<T>::value, scaled_t<T>>
floating_to_fixed(T floating, std::int16_t block_exponent)
{
  const auto s = sign(floating);
  const auto e = exponent(floating);
  const auto m = significand(floating);

  assert(block_exponent >= e);
  const auto shift = (float_traits<T>::exponent_bits - 1 - (block_exponent - e));
  scaled_t<T> r;
  if (-float_traits<T>::significand_bits > shift) {
    r = 0;
  } else {
    if (shift > 0) {
      r = m << shift;
    } else {
      r = m >> (-shift);
    }
  }
  if (s) {
    r |= 1ull << (sizeof(T) * CHAR_BIT - 1);
  }
  return r;
}

template<class F, class T>
FRSZ_ATTRIBUTES std::enable_if_t<std::is_floating_point<F>::value && std::is_integral<T>::value, F>
fixed_to_floating(T fixed, std::int16_t block_exponent)
{
  static_assert(sizeof(T) == sizeof(F));
  const auto f = fixed & (ones_s<T>::value >> 1);
  const auto z = xstd::countl_zero(f) - 1; // number of zeros after the sign bit
  const auto shift = -std::min(z, block_exponent + ebias_s<F>::value);

  const auto s = fixed & T{ 1 } << (sizeof(T) * CHAR_BIT - 1);
  const auto e = static_cast<std::uint64_t>(shift + block_exponent + ebias_s<F>::value)
                 << float_traits<F>::significand_bits;
  const auto mantissa_shift = (float_traits<F>::exponent_bits - 1 + shift);
  scaled_t<F> m;
  if (mantissa_shift > 0) {
    m = f >> mantissa_shift;
  } else {
    m = f << (-mantissa_shift);
  }
  m = m & (~(scaled_t<T>(1) << float_traits<F>::significand_bits)); // unset the implicit bit
  const scaled_t<T> r = s | e | m;
  return xstd::bit_cast<F>(r);
}

} // namespace fp
} // namespace detail

// Note: this only works for (a * b > 0) and b != 0
template<class T>
constexpr T
ceildiv(const T& a, const T& b)
{
  return a / b + (a % b != 0);
}

template<class T>
FRSZ_ATTRIBUTES T
abs(const T& val)
{
  return std::abs(val);
}

template<>
FRSZ_ATTRIBUTES float
abs(const float& val)
{
#if defined(__CUDA_ARCH__)
  return fabsf(val);
#else
  return std::abs(val);
#endif
}

template<>
FRSZ_ATTRIBUTES double
abs(const double& val)
{
#if defined(__CUDA_ARCH__)
  return fabs(val);
#else
  return std::abs(val);
#endif
}

// clang-format off
  /*
   * glossary
   *
   * total_elements  -- the total number of elements provided as input
   * data            -- pointer to the start of the data
   * compressed      -- pointer to the start of the compressed data
   * InputType       -- the unsigned integer type that fits the input type
   * OutputType      -- the unsigned integer type that has at least `bits` bits
   * ExpType         -- the type used to store the exponent
   *
   * max_exp_block_size -- the maximum exponent block size in elements
   * num_exp_blocks     -- the number of blocks that share a common exponent
   * exp_block_id       -- the block id of a block that share a common exponent
   * exp_block_elements -- the number of elements in this exp_block
   * exp_block_bytes    -- the number of bytes in a block that share a common exponent
   * exp_block_data_offset        -- what index in `data` does this exp_block start on?
   * scaled             -- temporary array storing scaled values for this block
   * exp_block_compressed   -- where compressed storage for a block that share a common exponent starts
   *
   * work_block_size -- the maximum work block size in elements
   * num_work_blocks     -- the number of work blocks in an exp_block
   * work_block_id       -- within an exp_block there are multiple work blocks, this is their id
   * work_block_elements -- the number of elements in a work block
   * work_block_bytes    -- the number of bytes used for a workblock; excludes the exponent
   *
   * output_bit_offset   -- what bit a work_block_element's bits should start on
   * output_byte_offset  -- what byte a work_block_element's byte should start on
   */
// clang-format on
template<int bits_per_value_, int max_exp_block_size_, class FpType_, class ExpType_>
struct frsz2_compressor
{
  constexpr static auto bits_per_value = bits_per_value_;
  constexpr static auto max_exp_block_size = max_exp_block_size_;
  using fp_type = FpType_;
  using exp_type = ExpType_;
  static_assert(bits_per_value <= sizeof(fp_type) * CHAR_BIT,
                "The number of bits per compressed value must be smaller (or equal to) the size of the "
                "original value type!");
  static_assert(0 < max_exp_block_size, "max_exp_block_size must be positive!");
  static_assert(4 <= bits_per_value,
                "Minimum bits per value is 4 (so at most 2 values share the same byte).");
  static_assert(bits_per_value <= 64, "maximum number for bits_per_value is 64.");

  using uint_compressed_type = detail::storage_t<bits_per_value>;
  using uint_fp_type = detail::scaled_t<fp_type>;

  static constexpr int uint_compressed_size_bit = sizeof(uint_compressed_type) * CHAR_BIT;
  static constexpr int compressed_block_size_byte =
    ceildiv<int>(max_exp_block_size * bits_per_value, CHAR_BIT) + sizeof(exp_type);

  static constexpr FRSZ_ATTRIBUTES std::size_t compute_compressed_memory_size_byte(
    std::size_t number_elements)
  {
    const std::size_t remainder = number_elements % max_exp_block_size;
    return (number_elements / max_exp_block_size) * compressed_block_size_byte +
           (remainder > 0) *
             (sizeof(exp_type) + ceildiv<std::size_t>(remainder * bits_per_value, uint_compressed_size_bit) *
                                   sizeof(uint_compressed_type));
  }

  static constexpr FRSZ_ATTRIBUTES uint_fp_type compressed_to_uint_fp(const uint_compressed_type& val)
  {
    return static_cast<uint_fp_type>(val) << (sizeof(uint_fp_type) * CHAR_BIT - bits_per_value);
  }

  static constexpr FRSZ_ATTRIBUTES uint_compressed_type uint_fp_to_compressed(const uint_fp_type& val)
  {
    return static_cast<uint_compressed_type>(val >> (sizeof(uint_fp_type) * CHAR_BIT - bits_per_value));
  }

  // This template argument is necessary in order to use SFINAE properly (otherwise, this results in a
  // compilation error)
  // TODO incorporate the shift and conversion to and from floating point in these functions
  template<int bits_per_value2 = bits_per_value>
  static constexpr FRSZ_ATTRIBUTES std::enable_if_t<uint_compressed_size_bit == bits_per_value2>
  write_shift_value(const int,
                    const uint_fp_type& input,
                    uint_compressed_type& first_val,
                    uint_compressed_type&)
  {
    static_assert(
      bits_per_value2 == bits_per_value,
      "This template parameter only exists to allow for SFINAE. Please don't change the default value.");
    first_val = uint_fp_to_compressed(input);
  }

  template<int bits_per_value2 = bits_per_value>
  static constexpr FRSZ_ATTRIBUTES std::enable_if_t<uint_compressed_size_bit == bits_per_value2, uint_fp_type>
  retrieve_shift_value(const int, const uint_compressed_type& first_val, const uint_compressed_type&)
  {
    static_assert(
      bits_per_value2 == bits_per_value,
      "This template parameter only exists to allow for SFINAE. Please don't change the default value.");
    return compressed_to_uint_fp(first_val);
  }

  // TODO check if UP when `second_val` references a value that is out of bound (but never used)
  template<int bits_per_value2 = bits_per_value>
  static constexpr FRSZ_ATTRIBUTES std::enable_if_t<uint_compressed_size_bit != bits_per_value2, uint_fp_type>
  retrieve_shift_value(const int bit_offset,
                       const uint_compressed_type& first_val,
                       const uint_compressed_type& second_val)
  {
    static_assert(
      bits_per_value2 == bits_per_value,
      "This template parameter only exists to allow for SFINAE. Please don't change the default value.");
    auto res = first_val >> bit_offset;
    res |= (uint_compressed_size_bit < bits_per_value2 + bit_offset)
             ? second_val << (uint_compressed_size_bit - bit_offset)
             : uint_compressed_type{};
    return compressed_to_uint_fp(res);
  }

  template<int bits_per_value2 = bits_per_value>
  static constexpr FRSZ_ATTRIBUTES std::enable_if_t<uint_compressed_size_bit != bits_per_value2>
  write_shift_value(const int bit_offset,
                    const uint_fp_type& input,
                    uint_compressed_type& first_val,
                    uint_compressed_type& second_val)
  {
    static_assert(
      bits_per_value2 == bits_per_value,
      "This template parameter only exists to allow for SFINAE. Please don't change the default value.");
    const auto converted_input = uint_fp_to_compressed(input);
    first_val = converted_input << bit_offset;
    second_val = (uint_compressed_size_bit < bits_per_value2 + bit_offset)
                   ? converted_input >> (uint_compressed_size_bit - bit_offset)
                   : uint_compressed_type{};
  }

// TODO remove majority of shared memory when bits_per_value is power of 2
// TODO maybe let every thread read only the data it needs to without putting it into shared memory
// TODO simplify decompression code significantly (maybe also the compression code)
#ifdef __CUDA_ARCH__
  /*
   * max_exp_block_size -- the maximum exponent block size in elements
   *
   * Threadblock : Exp_block_size is 1:1
   * exponent_block_idx must be identical for all threads in a thread block!
   */
  __device__ static void compress_gpu_function(const std::size_t exponent_block_idx,
                                               const fp_type fp_input_value,
                                               const uint64_t total_elements,
                                               uint8_t* const __restrict__ compressed)
  {
    constexpr auto min_exp_value =
      std::numeric_limits<typename detail::fp::float_traits<fp_type>::exponent_t>::min();

    const std::size_t num_exp_blocks = ceildiv<std::size_t>(total_elements, max_exp_block_size);
    const std::size_t total_compression_size = compute_compressed_memory_size_byte(total_elements);

    // TODO For multiple blocks per thread block, more shared memory is needed
    constexpr int required_shared_memory{ std::max<int>(
      max_exp_block_size * sizeof(exp_type), ceildiv<int>(max_exp_block_size * bits_per_value, CHAR_BIT)) };

    __shared__ volatile std::uint8_t shared_memory[required_shared_memory];
    // Since they are not used simultaneously, use the same shared memory for two purposes
    // FIXME could be UB since I write to shared_block_exponent and read from shared_memory!
    //       Maybe the volatile specifier prevents it from UB
    // Note: this should be legal as shared_memory is unsigned char, therefore, Type aliasing rules are
    //       followed
    auto shared_max_exponent =
      reinterpret_cast<volatile typename detail::fp::float_traits<fp_type>::exponent_t*>(shared_memory);
    auto shared_compressed = reinterpret_cast<volatile uint_compressed_type*>(shared_memory);

    // find the max exponent in the block to determine the bias
    shared_max_exponent[threadIdx.x] = detail::fp::exponent(fp_input_value);
    __syncthreads();
    // TODO make it a proper reduction with shuffles
    // TODO Shared memory usage could be eliminated when max_exp_block_size <=32
    // TODO specialize syncronization for max_exp_block_size <= 32 (subwarp-sync instead of thread block
    // sync)
    for (int i = detail::get_next_power_of_two_value(max_exp_block_size) / 2; i > 0; i >>= 1) {
      if (threadIdx.x < i) {
        const auto exp1 = shared_max_exponent[threadIdx.x];
        const auto exp2 =
          threadIdx.x + i < max_exp_block_size ? shared_max_exponent[threadIdx.x + i] : min_exp_value;
        shared_max_exponent[threadIdx.x] = std::max(exp1, exp2);
      }
      __syncthreads();
    }
    const exp_type max_exp{ shared_max_exponent[0] };

    // preform the scaling
    const auto exp_value_scaled = detail::fp::floating_to_fixed(fp_input_value, max_exp);
    static_assert(std::is_unsigned<decltype(exp_value_scaled)>::value,
                  "exp_value_scaled must be an unsigned type!");

    // compute the exp_block offset
    std::uint8_t* exp_block_compressed = compressed + exponent_block_idx * compressed_block_size_byte;

    if (threadIdx.x == 0) {
      memcpy(exp_block_compressed, &max_exp, sizeof(exp_type));
    }

    // at this point we have scaled values that we can encode
    const int output_bit_offset = (threadIdx.x * bits_per_value) % uint_compressed_size_bit;
    const int output_start_idx = (threadIdx.x * bits_per_value) / uint_compressed_size_bit;

    uint_compressed_type first_val;
    uint_compressed_type second_val;
    write_shift_value(output_bit_offset, exp_value_scaled, first_val, second_val);
    // Set shared memory to all zeros first:
    for (int i = threadIdx.x;
         i < ceildiv<int>(max_exp_block_size * bits_per_value, sizeof(uint_compressed_type) * CHAR_BIT);
         i += blockDim.x) {
      shared_compressed[i] = 0;
    }
    __syncthreads();
    // Note: it is possible to have a 3-way conflict per value (as long as bits_per_value >=4)
    //       For smaller bits_per_value, more synchronization would be necessary
    static_assert((uint_compressed_size_bit & (uint_compressed_size_bit - 1)) == 0,
                  "Bit size of the compressed uint must be a power of 2!");
    if (output_bit_offset < uint_compressed_size_bit / 2) {
      shared_compressed[output_start_idx] = first_val;
    }
    __syncthreads();
    if (bits_per_value != uint_compressed_size_bit) { // should be an if constexpr
      if (output_bit_offset >= uint_compressed_size_bit / 2) {
        shared_compressed[output_start_idx] |= first_val;
      }
      __syncthreads();
      // If the second_val is populated (otherwise, the default value for shared_compressed is 0 anyway):
      if (second_val != 0) {
        shared_compressed[output_start_idx + 1] |= second_val;
      }
      __syncthreads();
    }
    // if (blockIdx.x == 0) {
    //   printf("%d: %e / %.4x dissected as %.4x %.4x; shared_mem (idx %d, bit %d): %.4x %.4x\n",
    //          threadIdx.x,
    //          double(fp_input_value),
    //          int(exp_value_scaled),
    //          int(first_val),
    //          int(second_val),
    //          int(output_start_idx),
    //          int(output_bit_offset),
    //          int(shared_compressed[output_start_idx]),
    //          (uint_compressed_size_bit < output_bit_offset + bits_per_value)
    //            ? int(shared_compressed[output_start_idx + 1])
    //            : int(0));
    // }
    // Now, write out byte for byte since the `compressed` pointer might be not aligned for
    // uint_compressed_type
    std::uint8_t* compressed_output = exp_block_compressed + sizeof(exp_type);
    for (int i = threadIdx.x; i < ceildiv<int>(max_exp_block_size * bits_per_value, CHAR_BIT) &&
                              exponent_block_idx * compressed_block_size_byte + i < total_compression_size;
         i += blockDim.x) {
      compressed_output[i] = shared_memory[i];
    }
  }

  // blocks_per_tb is the number of compression blocks handled by a single thread block
  template<int blocks_per_tb>
  static __device__ fp_type decompress_gpu_function(const std::size_t exponent_block_idx,
                                                    const std::uint64_t total_elements,
                                                    std::uint8_t const* __restrict__ compressed)
  {
    static_assert(0 < max_exp_block_size && max_exp_block_size <= 1024,
                  "Requirement: 0 < max_exp_block_size <= 1024");

    // TODO figure out a better way to divide work instead of work_block_size
    // TODO make fool-proof alignment
    constexpr int byte_alignment_block_start =
      std::max(std::alignment_of<uint_compressed_type>::value, sizeof(exp_type));
    constexpr int shared_memory_per_exp_block_size_byte =
      byte_alignment_block_start + compressed_block_size_byte +
      (byte_alignment_block_start + compressed_block_size_byte) % byte_alignment_block_start;

    static_assert(blocks_per_tb <= 1 ||
                    shared_memory_per_exp_block_size_byte % byte_alignment_block_start == 0,
                  "Alignment for the next exponent must fit!");

    const int local_block_idx = threadIdx.x / max_exp_block_size;
    const int local_element_idx = threadIdx.x % max_exp_block_size;

    __shared__ std::uint8_t shared_mem[blocks_per_tb * shared_memory_per_exp_block_size_byte];

    const auto shared_block_exponent =
      reinterpret_cast<exp_type*>(shared_mem + local_block_idx * shared_memory_per_exp_block_size_byte);
    const auto shared_compressed = reinterpret_cast<uint_compressed_type*>(
      shared_mem + local_block_idx * shared_memory_per_exp_block_size_byte + byte_alignment_block_start);

    // TODO for a good device function, the amount of shared memory needs to be a parameter
    // TODO make the amount of processed exponents block another templated parameter
    const std::size_t total_compressed_memory_size = compute_compressed_memory_size_byte(total_elements);

    // TODO FIXME make memcpy for exponent and non-aligned memory accesses; proceed with aligned accesses to
    //            shared memory
    const std::uint8_t* const exp_block_compressed =
      compressed + (exponent_block_idx + local_block_idx) * compressed_block_size_byte;
    // recover the exponent
    if (local_element_idx == 0) {
      if ((exponent_block_idx + local_block_idx) * compressed_block_size_byte <
          total_compressed_memory_size) {
        memcpy(shared_block_exponent, exp_block_compressed, sizeof(exp_type));
      } else {
        *shared_block_exponent = 0;
      }
    }
    const std::uint8_t* const block_compressed_start = exp_block_compressed + sizeof(exp_type);

    for (int byte_idx = local_element_idx; byte_idx < ceildiv(max_exp_block_size * bits_per_value, CHAR_BIT);
         byte_idx += max_exp_block_size) {
      // Read everything into shared memory first
      shared_mem[local_block_idx * shared_memory_per_exp_block_size_byte + byte_alignment_block_start +
                 byte_idx] =
        (byte_idx <
         total_compressed_memory_size - (exponent_block_idx + local_block_idx) * compressed_block_size_byte)
          ? block_compressed_start[byte_idx]
          : std::uint8_t{};
    }
    __syncthreads();

    // recover the scaled values
    const int input_bit_offset = (local_element_idx * bits_per_value) % uint_compressed_size_bit;
    const int input_idx = (local_element_idx * bits_per_value) / uint_compressed_size_bit;
    const auto extracted_compressed_value =
      retrieve_shift_value(input_bit_offset, shared_compressed[input_idx], shared_compressed[input_idx + 1]);

    const auto output_val =
      detail::fp::fixed_to_floating<fp_type>(extracted_compressed_value, *shared_block_exponent);
    // if (blockIdx.x <= 1) {
    //   printf("%d-%d: %e / %.4x reconstructed from %.4x %.4x; shared_mem (idx %d, bit %d): %.4x %.4x\n",
    //          blockIdx.x,
    //          threadIdx.x,
    //          double(output_val),
    //          int(extracted_compressed_value),
    //          int(shared_compressed[input_idx]),
    //          int(shared_compressed[input_idx + 1]),
    //          int(input_idx),
    //          int(input_bit_offset),
    //          int(shared_compressed[input_idx]),
    //          (uint_compressed_size_bit < input_bit_offset + bits_per_value)
    //            ? int(shared_compressed[input_idx + 1])
    //            : int(0));
    // }
    // de-scale the values
    // if (threadIdx.x + exponent_block_idx * max_exp_block_size < total_elements) {
    // output[threadIdx.x + exponent_block_idx * max_exp_block_size] = output_val;
    //  printf("%d, %d (%d, %d): output[%d] = %e from 0x%llx as 0x%llx exp: %d; idx, offset: %d, %d\n",
    //         blockIdx.x,
    //         threadIdx.x,
    //         gridDim.x,
    //         blockDim.x,
    //         int(threadIdx.x + exponent_block_idx * max_exp_block_size),
    //         double(output_val),
    //         std::uint64_t(shared_compressed[input_idx]),
    //         std::uint64_t(current_compressed_value),
    //         int(*shared_block_exponent),
    //         int(input_idx),
    //         int(input_bit_offset));
    // }
    return output_val;
  }

#endif // __CUDA_ARCH__

  static int compress_cpu_impl(fp_type const* data,
                               const std::size_t total_elements,
                               std::uint8_t* compressed)
  {
    const std::size_t num_exp_blocks = ceildiv<std::size_t>(total_elements, max_exp_block_size);
    for (std::size_t exp_block_id = 0; exp_block_id < num_exp_blocks; exp_block_id++) {

      // how many elements to process in this block?
      const std::size_t exp_block_elements =
        std::min<std::size_t>(max_exp_block_size, total_elements - exp_block_id * max_exp_block_size);

      // find the max exponent in the block to determine the bias
      const std::size_t exp_block_data_offset = max_exp_block_size * exp_block_id;
      fp_type in_max = 0;
      for (size_t i = 0; i < exp_block_elements; ++i) {
        in_max = std::max(in_max, std::abs(data[i + exp_block_data_offset]));
      }
      const exp_type max_exp = detail::fp::exponent(in_max);

      // compute the exp_block offset
      uint8_t* exp_block_compressed = compressed + compressed_block_size_byte * exp_block_id;
      std::memcpy(exp_block_compressed, &max_exp, sizeof(exp_type));

      // at this point we have scaled values that we can encode

      const auto max_local_iterations =
        std::min<std::size_t>(max_exp_block_size, total_elements - exp_block_id * max_exp_block_size);
      uint_compressed_type overlap{};
      for (std::size_t local_idx = 0; local_idx < max_local_iterations; ++local_idx) {
        const std::size_t output_bit_offset = (local_idx * bits_per_value) % uint_compressed_size_bit;
        const std::size_t output_byte_offset =
          sizeof(exp_type) +
          ((local_idx * bits_per_value) / uint_compressed_size_bit) * sizeof(uint_compressed_type);
        const auto global_idx = local_idx + exp_block_id * max_exp_block_size;

        uint_compressed_type temp[2] = { 0, 0 };
        const auto to_store = detail::fp::floating_to_fixed(data[global_idx], max_exp);

        write_shift_value(output_bit_offset, to_store, temp[0], temp[1]);
        // std::cout << global_idx << ": " << std::showpos << std::setprecision(5) << data[global_idx]
        //           << " (max_e " << int(max_exp) << ") / " << std::hex << std::setw(2) << int(to_store)
        //           << " from " << std::setw(2) << int(temp[0]) << ' ' << std::setw(2) << int(temp[1])
        //           << "; bit " << std::dec << std::noshowbase << output_bit_offset << " byte "
        //           << output_byte_offset << '\n';

        if (uint_compressed_size_bit == bits_per_value) { // is pretty much an if constexpr
          memcpy(exp_block_compressed + output_byte_offset, temp, sizeof(uint_compressed_type));
        } else {
          if (output_bit_offset == 0) {
            overlap = temp[0];
            // clearly no need for temp[1]
          } else if (output_bit_offset + bits_per_value < uint_compressed_size_bit) {
            overlap |= temp[0];
            // also no need for temp[1] as everything still fits into the first value (temp[0])
          } else {
            overlap |= temp[0];
            memcpy(exp_block_compressed + output_byte_offset, &overlap, sizeof(uint_compressed_type));
            // Even if all the information is in temp[0], temp[1] is guaranteed to contain the value 0 (as
            // long as uint_compressed_size_bit != bits_per_value, which is guaranteed in this context)
            overlap = temp[1];
          }
        }
      }
      if (uint_compressed_size_bit != bits_per_value) { // is pretty much an if constexpr
        const std::size_t last_output_bit_offset =
          ((max_local_iterations - 1) * bits_per_value) % uint_compressed_size_bit;
        const std::size_t last_output_byte_offset =
          sizeof(exp_type) + (((max_local_iterations - 1) * bits_per_value) / uint_compressed_size_bit) *
                               sizeof(uint_compressed_type);
        // Write out the last overlap value if it hasn't previously and contains valuable information
        if (last_output_bit_offset + bits_per_value != uint_compressed_size_bit) {
          const auto actual_offset = (last_output_bit_offset + bits_per_value < uint_compressed_size_bit)
                                       ? last_output_byte_offset
                                       : last_output_byte_offset + sizeof(uint_compressed_type);
          memcpy(exp_block_compressed + actual_offset, &overlap, sizeof(uint_compressed_type));
        }
      }
    }
    return 0;
  }

  static int decompress_cpu_impl(fp_type* output,
                                 const std::uint64_t total_elements,
                                 std::uint8_t const* compressed)
  {
    // using InputType = detail::storage_t<bits_per_value>;
    // using int_output_type = detail::scaled_t<fp_type>;

    const std::size_t num_exp_blocks = ceildiv<std::size_t>(total_elements, max_exp_block_size);
    for (std::size_t exp_block_id = 0; exp_block_id < num_exp_blocks; exp_block_id++) {

      const std::uint8_t* exp_block_compressed = compressed + compressed_block_size_byte * exp_block_id;

      // recover the exponent
      exp_type block_exp = {};
      std::memcpy(&block_exp, exp_block_compressed, sizeof(exp_type));

      // recover the scaled values
      const std::size_t max_local_iterations =
        std::min<std::size_t>(max_exp_block_size, total_elements - exp_block_id * max_exp_block_size);
      uint_compressed_type tmp[2] = { 0, 0 };
      for (std::size_t local_idx = 0; local_idx < max_local_iterations; ++local_idx) {
        const int output_bit_offset = (local_idx * bits_per_value) % uint_compressed_size_bit;
        const int output_byte_offset =
          sizeof(exp_type) +
          ((local_idx * bits_per_value) / uint_compressed_size_bit) * sizeof(uint_compressed_type);
        const int copy_size = (bits_per_value + output_bit_offset > uint_compressed_size_bit)
                                ? 2 * sizeof(uint_compressed_type)
                                : sizeof(uint_compressed_type);
        std::memcpy(tmp, exp_block_compressed + output_byte_offset, copy_size);
        const auto local_val = retrieve_shift_value(output_bit_offset, tmp[0], tmp[1]);
        output[exp_block_id * max_exp_block_size + local_idx] =
          detail::fp::fixed_to_floating<fp_type>(local_val, block_exp);
      }
    }
    return 0;
  }

}; // struct frsz2_compressor

#if defined(__CUDACC__)
template<typename Frsz2Compressor>
__global__ void
compress_gpu(const typename Frsz2Compressor::fp_type* __restrict__ data,
             const std::size_t total_elements,
             std::uint8_t* const __restrict__ compressed)
{
  // requires that all threads of a thread block take the loop iteration, so there is no synchronization
  // problem
  constexpr std::size_t max_exp_blk_size = Frsz2Compressor::max_exp_block_size;
  for (std::size_t exp_blk_idx = blockIdx.x; exp_blk_idx < ceildiv(total_elements, max_exp_blk_size);
       exp_blk_idx += gridDim.x) {
    const auto global_idx = threadIdx.x + exp_blk_idx * max_exp_blk_size;

    Frsz2Compressor::compress_gpu_function(exp_blk_idx,
                                           global_idx < total_elements ? data[global_idx]
                                                                       : typename Frsz2Compressor::fp_type{},
                                           total_elements,
                                           compressed);
  }
}

template<typename Frsz2Compressor, int blocks_per_tb>
__global__ void
decompress_gpu(typename Frsz2Compressor::fp_type* const __restrict__ output,
               const std::size_t total_elements,
               const std::uint8_t* const __restrict__ compressed)
{
  constexpr std::size_t max_exp_blk_size = Frsz2Compressor::max_exp_block_size;
  for (std::size_t exp_blk_idx = blockIdx.x * blocks_per_tb;
       exp_blk_idx < ceildiv(total_elements, max_exp_blk_size);
       exp_blk_idx += gridDim.x * blocks_per_tb) {
    const auto global_idx = threadIdx.x + exp_blk_idx * max_exp_blk_size;

    const auto out =
      Frsz2Compressor::decompress_gpu_function<blocks_per_tb>(exp_blk_idx, total_elements, compressed);
    if (global_idx < total_elements) {
      output[global_idx] = out;
      //   printf("%d-%d: %lld is fine\n", blockIdx.x, threadIdx.x, global_idx);
      // } else {
      //   printf("%d-%d: %lld is NOT fine because\n", blockIdx.x, threadIdx.x, global_idx);
    }
  }
}

#endif // __CUDACC__

template<int... values>
struct int_list_t
{};

#define CREATE_FRSZ2_DISPATCH(_DISP_NAME, _FUNCTION)                                                         \
  template<int bits_per_value, class FpType, class ExpType, class... Args>                                   \
  void _DISP_NAME##_impl(int_list_t<>, const int actual_exp_val, Args&&...)                                  \
  {                                                                                                          \
    throw std::runtime_error("Unsupported exponent block size: " + std::to_string(actual_exp_val));          \
  }                                                                                                          \
  template<int bits_per_value,                                                                               \
           class FpType,                                                                                     \
           class ExpType,                                                                                    \
           int curr_exp_size,                                                                                \
           int... exp_block_sizes,                                                                           \
           class... Args>                                                                                    \
  void _DISP_NAME##_impl(                                                                                    \
    int_list_t<curr_exp_size, exp_block_sizes...>, const int actual_exp_val, Args&&... args)                 \
  {                                                                                                          \
    if (curr_exp_size == actual_exp_val) {                                                                   \
      frsz2_compressor<bits_per_value, curr_exp_size, FpType, ExpType>::_FUNCTION(                           \
        std::forward<Args>(args)...);                                                                        \
    } else {                                                                                                 \
      _DISP_NAME##_impl<bits_per_value, FpType, ExpType>(                                                    \
        int_list_t<exp_block_sizes...>{}, actual_exp_val, std::forward<Args>(args)...);                      \
    }                                                                                                        \
  }                                                                                                          \
  template<class FpType, class ExpType, int... exp_block_sizes, class... Args>                               \
  void _DISP_NAME(                                                                                           \
    int_list_t<>, const int actual_bits_val, int_list_t<exp_block_sizes...>, const int, Args&&...)           \
  {                                                                                                          \
    throw std::runtime_error("Unsupported number of bits: " + std::to_string(actual_bits_val));              \
  }                                                                                                          \
  template<class FpType,                                                                                     \
           class ExpType,                                                                                    \
           int curr_bit_value,                                                                               \
           int... bit_values,                                                                                \
           int... exp_block_sizes,                                                                           \
           class... Args>                                                                                    \
  void _DISP_NAME(int_list_t<curr_bit_value, bit_values...>,                                                 \
                  int actual_bits_val,                                                                       \
                  int_list_t<exp_block_sizes...> exp_list,                                                   \
                  int actual_exp_val,                                                                        \
                  Args&&... args)                                                                            \
  {                                                                                                          \
    if (curr_bit_value == actual_bits_val) {                                                                 \
      _DISP_NAME##_impl<curr_bit_value, FpType, ExpType>(                                                    \
        exp_list, actual_exp_val, std::forward<Args>(args)...);                                              \
    } else {                                                                                                 \
      _DISP_NAME<FpType, ExpType>(int_list_t<bit_values...>{},                                               \
                                  actual_bits_val,                                                           \
                                  exp_list,                                                                  \
                                  actual_exp_val,                                                            \
                                  std::forward<Args>(args)...);                                              \
    }                                                                                                        \
  }

CREATE_FRSZ2_DISPATCH(dispatch_frsz2_compression, compress_cpu_impl)
CREATE_FRSZ2_DISPATCH(dispatch_frsz2_decompression, decompress_cpu_impl)

} // namespace frsz

#endif
