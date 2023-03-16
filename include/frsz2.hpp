#ifndef FRSZ_FRSZ2_CUH
#define FRSZ_FRSZ2_CUH

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

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

// TODO call device function internally
template<class T>
constexpr FRSZ_ATTRIBUTES std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value, int>
countl_zero(T val) noexcept
{
  return val == 0 ? sizeof(T) * CHAR_BIT : countl_zero(val >> 1) - 1;
}

} // namespace xstd

namespace detail {

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

/**
 * \param[in] in pointer to the byte containing the bytes to shift
 * \param[in] bits how many bits to shift left
 * \param[in] offset what bit does the number start on?
 */
// TODO FIXME likely little vs big endian problem
// TODO make `bits` a template parameter
template<class OutputType, class InputType>
FRSZ_ATTRIBUTES OutputType
shift_left(InputType const* in, std::uint8_t bits, std::uint8_t offset)
{
  assert(offset < CHAR_BIT);
  assert(bits <= sizeof(InputType) * CHAR_BIT);
  OutputType result = in[0];
  result <<= offset;
  if (bits + offset > sizeof(InputType) * CHAR_BIT) {
    const std::uint8_t remaining = bits - (sizeof(InputType) * CHAR_BIT - offset);
    OutputType remaining_bits = ((in[1] >> (sizeof(InputType) * CHAR_BIT - remaining)) &
                                 (detail::ones_s<InputType>::value >> remaining));
    remaining_bits <<= (sizeof(InputType) * CHAR_BIT - bits);
    result |= remaining_bits;
  }
  if (sizeof(OutputType) > sizeof(InputType)) {
    result <<= CHAR_BIT * (sizeof(OutputType) - sizeof(InputType));
  }
  return result;
}

template<std::uint8_t bits_per_value, int max_exp_block_size, class ExpType>
constexpr FRSZ_ATTRIBUTES std::size_t
compute_compressed_memory_size_byte(std::size_t number_elements)
{
  constexpr std::size_t exp_block_byte =
    ceildiv<std::size_t>(max_exp_block_size * bits_per_value, CHAR_BIT) + sizeof(ExpType);
  const std::size_t remainder = number_elements % max_exp_block_size;
  return (number_elements / max_exp_block_size) * exp_block_byte +
         (remainder > 0) * (sizeof(ExpType) + ceildiv<std::size_t>(remainder * bits_per_value, CHAR_BIT));
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
   * max_work_block_size -- the maximum work block size in elements
   * num_work_blocks     -- the number of work blocks in an exp_block
   * work_block_id       -- within an exp_block there are multiple work blocks, this is their id
   * work_block_elements -- the number of elements in a work block
   * work_block_bytes    -- the number of bytes used for a workblock; excludes the exponent
   *
   * output_bit_offset   -- what bit a work_block_element's bits should start on
   * output_byte_offset  -- what byte a work_block_element's byte should start on
   */
// clang-format on
/*
 * max_exp_block_size -- the maximum exponent block size in elements
 * subwarp_size -- number of threads that are bundled into a subwarp to perform the decompression in lock-step
 *
 * Threadblock : Exp_block_size is 1:1
 */
template<std::uint8_t bits_per_value, int max_exp_block_size, int subwarp_size, class T, class ExpType>
__global__ void
decompress_gpu_impl(T* output, const std::uint64_t total_elements, std::uint8_t const* compressed)
{
  static_assert((subwarp_size * bits_per_value) % CHAR_BIT == 0, "the work blocks must be byte aligned");
  static_assert(subwarp_size <= max_exp_block_size,
                "the exp block must be as large or larger than the work_block");
  static_assert(0 < subwarp_size && subwarp_size <= 32, "Subwarp size must be a valid subwarp size.");
  static_assert((subwarp_size & (subwarp_size - 1)) == 0, "Subwarp size must be a power of 2.");

  using int_compressed_type = detail::storage_t<bits_per_value>;
  using int_output_type = detail::scaled_t<T>;

  // TODO figure out a better way to divide work instead of subwarp_size
  constexpr int exp_block_bytes{ ceildiv(max_exp_block_size * bits_per_value, CHAR_BIT) + sizeof(ExpType) };
  // TODO make fool-proof alignment
  constexpr int byte_alignment_block_start =
    std::max(std::alignment_of<int_compressed_type>::value, sizeof(ExpType));

  __shared__ std::uint8_t shared_mem[exp_block_bytes + byte_alignment_block_start];
  auto shared_block_exponent = reinterpret_cast<ExpType*>(shared_mem);
  auto shared_compressed = reinterpret_cast<int_compressed_type*>(shared_mem + byte_alignment_block_start);

  // TODO for a good device function, the amount of shared memory needs to be a parameter
  // TODO make the amount of processed exponents block another templated parameter
  const std::size_t num_exp_blocks = ceildiv<std::size_t>(total_elements, max_exp_block_size);
  const std::size_t compressed_memory_size =
    compute_compressed_memory_size_byte<bits_per_value, max_exp_block_size, ExpType>(total_elements);
  // For now, process 1 exp block per thread block
  for (std::size_t exp_block_id = blockIdx.x; exp_block_id < num_exp_blocks; exp_block_id += gridDim.x) {

    // TODO FIXME make memcpy for exponent and non-aligned memory accesses; proceed with aligned accesses to
    //            shared memory
    const std::uint8_t* exp_block_compressed = compressed + exp_block_id * exp_block_bytes;
    // recover the exponent
    if (threadIdx.x == 0) {
      memcpy(shared_block_exponent, exp_block_compressed, sizeof(ExpType));
    }
    const std::uint8_t* block_compressed_start = exp_block_compressed + sizeof(ExpType);

    for (int byte_idx = threadIdx.x; byte_idx < ceildiv(max_exp_block_size * bits_per_value, CHAR_BIT);
         byte_idx += blockDim.x) {
      // Read everything into shared memory first
      shared_mem[byte_alignment_block_start + byte_idx] =
        (byte_idx < compressed_memory_size - exp_block_id * exp_block_bytes)
          ? block_compressed_start[byte_idx]
          : std::uint8_t{};
    }
    __syncthreads();

    // recover the scaled values
    const std::size_t input_bit_offset =
      (threadIdx.x * bits_per_value) % (CHAR_BIT * sizeof(int_compressed_type));
    const std::size_t input_idx = (threadIdx.x * bits_per_value) / (CHAR_BIT * sizeof(int_compressed_type));
    const auto current_compressed_value =
      shift_left<int_output_type>(shared_compressed + input_idx, bits_per_value, input_bit_offset);

    const auto output_val =
      detail::fp::fixed_to_floating<T>(current_compressed_value, *shared_block_exponent);
    // de-scale the values
    if (threadIdx.x + exp_block_id * max_exp_block_size < total_elements) {
      output[threadIdx.x + exp_block_id * max_exp_block_size] = output_val;
      // printf("%d, %d (%d, %d): output[%d] = %e from 0x%llx as 0x%llx exp: %d; idx, offset: %d, %d\n",
      //        blockIdx.x,
      //        threadIdx.x,
      //        gridDim.x,
      //        blockDim.x,
      //        int(threadIdx.x + exp_block_id * max_exp_block_size),
      //        double(output_val),
      //        std::uint64_t(shared_compressed[input_idx]),
      //        std::uint64_t(current_compressed_value),
      //        int(*shared_block_exponent),
      //        int(input_idx),
      //        int(input_bit_offset));
    }
  }
}

template<std::uint8_t bits, int max_exp_block_size, int max_work_block_size, class T, class ExpType>
int
decompress_cpu_impl(T* output, const std::uint64_t total_elements, std::uint8_t const* compressed)
{
  static_assert((max_work_block_size * bits) % CHAR_BIT == 0, "the work blocks must be byte aligned");
  static_assert(max_work_block_size <= max_exp_block_size,
                "the exp block must be as large or larger than the work_block");
  static_assert(0 < max_work_block_size && max_work_block_size <= 32,
                "Subwarp size must be a valid subwarp size.");
  static_assert((max_work_block_size & (max_work_block_size - 1)) == 0, "Subwarp size must be a power of 2.");

  using InputType = detail::storage_t<bits>;
  using int_output_type = detail::scaled_t<T>;

  const std::size_t num_exp_blocks = ceildiv<std::size_t>(total_elements, max_exp_block_size);
  const std::size_t exp_block_bytes = ceildiv(max_exp_block_size * bits, CHAR_BIT) + sizeof(ExpType);
  for (std::size_t exp_block_id = 0; exp_block_id < num_exp_blocks; exp_block_id++) {

    const std::size_t exp_block_elements =
      std::min<std::size_t>(max_exp_block_size, total_elements - exp_block_id * max_exp_block_size);
    const std::size_t num_work_blocks = ceildiv<std::size_t>(exp_block_elements, max_work_block_size);
    const std::size_t work_block_bytes = ceildiv(max_work_block_size * bits, CHAR_BIT);
    const std::size_t exp_block_data_offset = max_exp_block_size * exp_block_id;
    const std::uint8_t* exp_block_compressed = compressed + exp_block_bytes * exp_block_id;

    // recover the exponent
    ExpType block_exp = {};
    std::memcpy(&block_exp, exp_block_compressed, sizeof(ExpType));

    // recover the scaled values
    std::vector<detail::scaled_t<T>> exp_block_scaled(max_exp_block_size);
    for (std::size_t work_block_id = 0; work_block_id < num_work_blocks; ++work_block_id) {
      const std::size_t work_block_elements =
        std::min<std::size_t>(max_work_block_size, exp_block_elements - work_block_id * max_work_block_size);
      std::size_t output_bit_offset = 0;
      std::size_t output_byte_offset = sizeof(ExpType) + work_block_id * work_block_bytes;
      for (std::size_t i = 0; i < work_block_elements; ++i) {
        InputType tmp[2] = { 0, 0 };
        const std::uint16_t copy_size = (bits + output_bit_offset > CHAR_BIT * sizeof(InputType))
                                          ? 2 * sizeof(InputType)
                                          : sizeof(InputType);
        const std::uint16_t safe_copy_size =
          std::min<std::uint16_t>(exp_block_bytes - output_byte_offset, copy_size);
        std::memcpy(tmp, exp_block_compressed + output_byte_offset, safe_copy_size);
        exp_block_scaled[i + work_block_id * max_work_block_size] =
          shift_left<int_output_type>(tmp, bits, output_bit_offset);

        output_byte_offset += (output_bit_offset + bits) / CHAR_BIT;
        output_bit_offset = (output_bit_offset + bits) % CHAR_BIT;
      }
    }

    // de-scale the values
    for (std::size_t i = 0; i < exp_block_elements; ++i) {
      output[i + exp_block_data_offset] = detail::fp::fixed_to_floating<T>(exp_block_scaled[i], block_exp);
    }
  }

  return 0;
}

template<std::uint8_t bits_per_value, int max_exp_block_size, int subwarp_size, class T, class ExpType>
__global__ void
compress_gpu_impl(T const* data, const uint64_t total_elements, uint8_t* compressed)
{
  static_assert((subwarp_size * bits_per_value) % CHAR_BIT == 0, "the work blocks must be byte aligned");
  static_assert(subwarp_size <= max_exp_block_size,
                "the exp block must be as large or larger than the work_block");
  static_assert(0 < subwarp_size && subwarp_size <= 32, "Subwarp size must be a valid subwarp size.");
  static_assert((subwarp_size & (subwarp_size - 1)) == 0, "Subwarp size must be a power of 2.");

  using InputType = detail::scaled_t<T>;
  using OutputType = detail::storage_t<bits_per_value>;

  constexpr int output_size_bit = sizeof(OutputType) * CHAR_BIT;

  constexpr auto min_exp_value = std::numeric_limits<typename detail::fp::float_traits<T>::exponent_t>::min();

  const std::size_t num_exp_blocks = ceildiv<std::size_t>(total_elements, max_exp_block_size);
  const std::size_t exp_block_bytes =
    ceildiv(max_exp_block_size * bits_per_value, CHAR_BIT) + sizeof(ExpType);
  const std::size_t total_compression_size =
    compute_compressed_memory_size_byte<bits_per_value, max_exp_block_size, ExpType>(total_elements);

  // TODO For multiple blocks per thread block, more shared memory is needed
  constexpr int required_shared_memory{ std::max<int>(
    max_exp_block_size * sizeof(ExpType), ceildiv<int>(max_exp_block_size * bits_per_value, CHAR_BIT)) };

  __shared__ volatile std::uint8_t shared_memory[required_shared_memory];
  // Since they are not used simultaneously, use the same shared memory for two purposes
  // FIXME could be UB since I write to shared_block_exponent and read from shared_memory!
  //       Maybe the volatile specifier prevents it from UB
  // Note: this should be legal as shared_memory is unsigned char, therefore, Type aliasing rules are followed
  auto shared_max_exponent =
    reinterpret_cast<volatile typename detail::fp::float_traits<T>::exponent_t*>(shared_memory);
  auto shared_compressed = reinterpret_cast<volatile OutputType*>(shared_memory);
  for (std::size_t exp_block_id = blockIdx.x; exp_block_id < num_exp_blocks; exp_block_id += gridDim.x) {

    const std::size_t global_idx = threadIdx.x + exp_block_id * max_exp_block_size;

    const auto fp_input_value = global_idx < total_elements ? data[global_idx] : 0;
    // find the max exponent in the block to determine the bias
    shared_max_exponent[threadIdx.x] = detail::fp::exponent(fp_input_value);
    __syncthreads();
    // TODO make it a proper reduction with shuffles
    // TODO Shared memory usage could be eliminated when max_exp_block_size <=32
    // TODO specialize syncronization for max_exp_block_size <= 32 (subwarp-sync instead of thread block sync)
    for (int i = ceildiv(max_exp_block_size, 2); i > 0; i >>= 1) {
      if (threadIdx.x < i) {
        const auto exp1 = shared_max_exponent[threadIdx.x];
        const auto exp2 =
          threadIdx.x + i < max_exp_block_size ? shared_max_exponent[threadIdx.x + i] : min_exp_value;
        shared_max_exponent[threadIdx.x] = std::max(exp1, exp2);
      }
      __syncthreads();
    }
    const ExpType max_exp{ shared_max_exponent[0] };

    // preform the scaling
    // TODO take care of left shifts as well!
    const InputType exp_value_scaled =
      detail::fp::floating_to_fixed(fp_input_value, max_exp) >> sizeof(InputType) * CHAR_BIT - bits_per_value;
    static_assert(std::is_unsigned<decltype(exp_value_scaled)>::value,
                  "exp_value_scaled must be an unsigned type!");

    // compute the exp_block offset
    std::uint8_t* exp_block_compressed = compressed + exp_block_id * exp_block_bytes;

    // bzero(exp_block_compressed, exp_block_bytes); // same as memset(dest, 0x00, size);
    if (threadIdx.x == 0) {
      memcpy(exp_block_compressed, &max_exp, sizeof(ExpType));
    }

    // at this point we have scaled values that we can encode
    const std::size_t output_bit_offset = (threadIdx.x * bits_per_value) % output_size_bit;
    const std::size_t output_start_idx = (threadIdx.x * bits_per_value) / output_size_bit;
    OutputType first_val = exp_value_scaled >> output_bit_offset;
    /* Small example with the shifts:
     * OutputType: 32bit;   bits_per_val: 20bit;    bit_offset: 30
     * first_val = val >> 30
     * second_val = val << 12 + 32-30 = val << 14
     */
    // 2nd left shift: output_size_bit - bits_per_value + (output_size_bit - output_bit_offset)
    //                 = 2*output_size_bit - bits_per_value - output_bit_offset
    OutputType second_val = (output_size_bit < output_bit_offset + bits_per_value)
                              ? exp_value_scaled << 2 * output_size_bit - bits_per_value - output_bit_offset
                              : OutputType{};
    // Note: it is possible to have a 3-way conflict per value
    if (output_bit_offset < output_size_bit / 2) {
      shared_compressed[output_start_idx] = first_val;
    }
    __syncthreads();
    if (bits_per_value != output_size_bit) {
      if (output_bit_offset >= output_size_bit / 2) {
        shared_compressed[output_start_idx] |= first_val;
      }
      __syncthreads();
      // If the second_val is populated:
      if (second_val != 0) {
        shared_compressed[output_start_idx + 1] |= second_val;
      }
      __syncthreads();
    }
    // Now, write out byte for byte since the `compressed` pointer might be not aligned for OutputType
    std::uint8_t* compressed_output = exp_block_compressed + sizeof(ExpType);
    for (int i = threadIdx.x; i < ceildiv<int>(max_exp_block_size * bits_per_value, CHAR_BIT) &&
                              exp_block_id * exp_block_bytes + i < total_compression_size;
         i += blockDim.x) {
      compressed_output[i] = shared_memory[i];
    }
  }
}

template<std::uint8_t bits_per_value, int max_exp_block_size, int max_work_block_size, class T, class ExpType>
int
compress_cpu_impl(T const* data, const uint64_t total_elements, uint8_t* compressed)
{
  static_assert((max_work_block_size * bits_per_value) % CHAR_BIT == 0,
                "the work blocks must be byte aligned");
  static_assert(max_work_block_size <= max_exp_block_size,
                "the exp block must be as large or larger than the work_block");
  static_assert(0 < max_work_block_size && max_work_block_size <= 32,
                "Subwarp size must be a valid subwarp size.");
  static_assert((max_work_block_size & (max_work_block_size - 1)) == 0, "Subwarp size must be a power of 2.");

  using InputType = detail::scaled_t<T>;
  using OutputType = detail::storage_t<bits_per_value>;

  const size_t num_exp_blocks = ceildiv<std::size_t>(total_elements, max_exp_block_size);
  const size_t exp_block_bytes = ceildiv(max_exp_block_size * bits_per_value, 8) + sizeof(ExpType);
  for (size_t exp_block_id = 0; exp_block_id < num_exp_blocks; exp_block_id++) {

    // how many elements to process in this block?
    const size_t exp_block_elements =
      std::min<std::size_t>(max_exp_block_size, total_elements - exp_block_id * max_exp_block_size);

    // find the max exponent in the block to determine the bias
    const size_t exp_block_data_offset = max_exp_block_size * exp_block_id;
    T in_max = 0;
    for (size_t i = 0; i < exp_block_elements; ++i) {
      in_max = std::max(in_max, std::fabs(data[i + exp_block_data_offset]));
    }
    ExpType max_exp = detail::fp::exponent(in_max);

    // preform the scaling
    std::vector<detail::scaled_t<T>> exp_block_scaled(max_exp_block_size);
    for (size_t i = 0; i < exp_block_elements; ++i) {
      exp_block_scaled[i] = detail::fp::floating_to_fixed(data[i + exp_block_data_offset], max_exp);
    }

    // compute the exp_block offset
    uint8_t* exp_block_compressed = compressed + exp_block_bytes * exp_block_id;
    bzero(exp_block_compressed, exp_block_bytes);
    memcpy(exp_block_compressed, &max_exp, sizeof(ExpType));

    // at this point we have scaled values that we can encode

    const size_t num_work_blocks = ceildiv<std::size_t>(exp_block_elements, max_work_block_size);
    const size_t work_block_bytes = ceildiv(max_work_block_size * bits_per_value, 8);

    for (size_t work_block_id = 0; work_block_id < num_work_blocks; ++work_block_id) {
      const size_t work_block_elements =
        std::min<std::size_t>(max_work_block_size, exp_block_elements - work_block_id * max_work_block_size);
      size_t output_bit_offset = 0;
      size_t output_byte_offset = sizeof(ExpType) + work_block_id * work_block_bytes;
      for (size_t i = 0; i < work_block_elements; ++i) {
        const detail::scaled_t<OutputType> to_store =
          exp_block_scaled[i + work_block_id * max_work_block_size] >>
          (sizeof(InputType) * 8 - bits_per_value);

        // we do this dance to avoid an unaligned load
        uint8_t* const out = exp_block_compressed + output_byte_offset;
        uint8_t const copy_size =
          ((bits_per_value + output_bit_offset > 8 * sizeof(OutputType)) ? 2 * sizeof(OutputType)
                                                                         : sizeof(OutputType));
        uint8_t const safe_copy_size = std::min<uint8_t>(copy_size, exp_block_bytes - output_byte_offset);
        OutputType temp[2] = { 0, 0 };

        memcpy(temp, out, safe_copy_size);

        // write everything from offset to output_type boundary
        uint8_t first_output_shift;
        uint8_t first_output_bits;
        if (bits_per_value + output_bit_offset > 8 * sizeof(OutputType)) {
          first_output_shift = 0;
          first_output_bits = 8 * sizeof(OutputType) - output_bit_offset;
        } else {
          first_output_shift = 8 * sizeof(OutputType) - (bits_per_value + output_bit_offset);
          first_output_bits = bits_per_value;
        }

        const OutputType first_store =
          (to_store >> (bits_per_value - first_output_bits) &
           (detail::ones_s<OutputType>::value >> (8 * sizeof(OutputType) - first_output_bits)))
          << first_output_shift;
        temp[0] |= first_store;

        // if there are leftovers, write those to the high-order bytes
        if (bits_per_value + output_bit_offset > 8 * sizeof(OutputType)) {
          const uint8_t remaining = bits_per_value - first_output_bits;
          const uint8_t second_shift = 8 * sizeof(OutputType) - remaining;
          const OutputType second_store =
            (to_store & (detail::ones_s<OutputType>::value >> (8 * sizeof(OutputType) - remaining)))
            << second_shift;
          temp[1] |= second_store;
        }

        // copy it back to avoid an unaligned store
        memcpy(out, temp, safe_copy_size);

        output_byte_offset += (bits_per_value + output_bit_offset) / 8;
        output_bit_offset = (output_bit_offset + bits_per_value) % 8;
      }
    }
  }

  return 0;
}

} // namespace frsz

#endif
