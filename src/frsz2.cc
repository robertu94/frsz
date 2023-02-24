#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"
#include <cmath>

#define runtime_assert(msg, ...)                                                                             \
  if (!(__VA_ARGS__))                                                                                        \
    throw std::runtime_error(std::string(msg) + #__VA_ARGS__);

namespace libpressio {
namespace frsz2_ns {

template<class T>
using scaled_t = std::conditional_t<
  sizeof(T) == 8,
  uint64_t,
  std::conditional_t<sizeof(T) == 4, uint32_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>>;
template<size_t N>
using storage_t = std::conditional_t<
  (N <= 8),
  uint8_t,
  std::conditional_t<(N <= 16), uint16_t, std::conditional_t<(N <= 32), uint32_t, uint64_t>>>;
template<class T>
inline constexpr const int expbits = static_cast<int>(
  8 * sizeof(uint32_t) - std::countl_zero(static_cast<uint32_t>(std::numeric_limits<T>::max_exponent)));
template<class T>
inline constexpr const int ebias = ((1 << (expbits<T> - 1)) - 1);

template<class T>
struct ones_t;

template<>
struct ones_t<uint8_t>
{
  static constexpr uint8_t value = 0xFF;
};
template<>
struct ones_t<uint16_t>
{
  static constexpr uint16_t value = 0xFFFF;
};
template<>
struct ones_t<uint32_t>
{
  static constexpr uint32_t value = 0xFFFFFFFF;
};
template<>
struct ones_t<uint64_t>
{
  static constexpr uint64_t value = 0xFFFFFFFFFFFFFFFF;
};
template<>
struct ones_t<int8_t>
{
  static constexpr uint8_t value = 0xFF;
};
template<>
struct ones_t<int16_t>
{
  static constexpr uint16_t value = 0xFFFF;
};
template<>
struct ones_t<int32_t>
{
  static constexpr uint32_t value = 0xFFFFFFFF;
};
template<>
struct ones_t<int64_t>
{
  static constexpr uint64_t value = 0xFFFFFFFFFFFFFFFF;
};

namespace fp {
    template <class T>
    struct float_traits {
    };
    template <>
    struct float_traits<float> {
        using sign_t = bool;
        using mantissa_t = int8_t;
        using significand_t = uint32_t;
        constexpr static int64_t sign_bits = 1;
        constexpr static int64_t exponent_bits = 8;
        constexpr static int64_t significand_bits = 23;
        
    };
    template <>
    struct float_traits<double> {
        using sign_t = bool;
        using mantissa_t = int16_t;
        using significand_t = uint64_t;
        constexpr static int64_t sign_bits = 1;
        constexpr static int64_t exponent_bits = 11;
        constexpr static int64_t significand_bits = 52;
    };

    template <class T>
    typename float_traits<T>::mantissa_t exponent(T f) {
        constexpr size_t mantissa_shift = float_traits<T>::significand_bits + float_traits<T>::sign_bits;  
        return (std::bit_cast<scaled_t<T>>(f)>>float_traits<T>::significand_bits & (ones_t<scaled_t<T>>::value >> mantissa_shift)) - ebias<T>;
    }
    template <class T>
    typename float_traits<T>::significand_t significand(T f) {
        scaled_t<T> s = std::bit_cast<scaled_t<T>>(f) & (ones_t<scaled_t<T>>::value >> (sizeof(T)*8 -float_traits<T>::significand_bits));
        if(!(exponent(f) == -ebias<T>)) {
            //is not subnormal, add leading 1 bit
            s |= 1ull<<float_traits<T>::significand_bits;
        }
        return s;
    }
    template <class T>
    typename float_traits<T>::sign_t sign(T f) {
        return (std::bit_cast<scaled_t<T>>(f) & (1ull<<(8*sizeof(T)-1)));
    }
    inline constexpr bool positive = false;
    inline constexpr bool negative = true;

    template <class T>
    int16_t is_normal(T f) {
        if(!(exponent(f) == -ebias<T>)) {
            return 1;
        }
        return 0;
    }

    template <class T>
    scaled_t<T> floating_to_fixed(T floating, int16_t block_exponent) {
        auto s = sign(floating);
        auto e = exponent(floating);
        auto m = significand(floating);

        assert(block_exponent >= e);
        auto shift = (float_traits<T>::exponent_bits - 1 - (block_exponent - e));
        scaled_t<T> r;
        if(-float_traits<T>::significand_bits > shift) {
            r = 0;
        } else {
            if(shift > 0) {
                r = m<<shift;
            } else {
                r = m>>(-shift);
            }
        }
        if(s) {
            r |= 1ull<<(sizeof(T)*8 -1);
        }
        return r;
    }
    template<class F, class T>
    F fixed_to_floating(T fixed, int16_t block_exponent){
        static_assert(sizeof(T) == sizeof(F));
        auto f = fixed & (ones_t<T>::value >> 1);
        auto z = std::countl_zero(f) - 1; //number of zeros after the sign bit
        auto shift = - std::min(z, block_exponent + ebias<F>);

        auto s = fixed & 1ull<<(sizeof(T)*8 - 1);
        auto e = static_cast<uint64_t>(shift + block_exponent + ebias<F>) << float_traits<F>::significand_bits;
        auto mantissa_shift = (float_traits<F>::exponent_bits - 1 + shift);
        scaled_t<F> m;
        if(mantissa_shift > 0) {
            m = f >> mantissa_shift;
        } else {
            m = f << (-mantissa_shift);
        }
        m = m & (~(scaled_t<T>(1)<<float_traits<F>::significand_bits)); //unset the implicit bit
        scaled_t<T> r = s | e | m;
        return std::bit_cast<F>(r);
    }
}


constexpr int
ceildiv(int a, int b)
{
  if (a % b == 0)
    return a / b;
  else
    return a / b + 1;
}


/**
 * \param[in] in pointer to the byte containing the bytes to shift
 * \param[in] bits how many bits to shift left
 * \param[in] offset what bit does the number start on?
 */
template<class OutputType, class InputType>
OutputType
shift_left(InputType const* in, uint8_t bits, uint8_t offset)
{
  assert(offset < 8);
  assert(bits <= sizeof(InputType) * 8);
  OutputType result = in[0];
  result <<= offset;
  if (bits + offset > sizeof(InputType) * 8) {
    const uint8_t remaining = bits - (sizeof(InputType) * 8 - offset);
    OutputType remaining_bits =
      ((in[1] >> (sizeof(InputType) * 8 - remaining)) & (ones_t<InputType>::value >> remaining));
    remaining_bits <<= (sizeof(InputType) * 8 - bits);
    result |= remaining_bits;
  }
  if (sizeof(OutputType) > sizeof(InputType)) {
    result <<= 8 * (sizeof(OutputType) - sizeof(InputType));
  }
  return result;
}



class frsz2_compressor_plugin : public libpressio_compressor_plugin
{
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "frsz2:bits", bits);
    set(options, "frsz2:max_exp_block_size", max_exp_block_size);
    set(options, "frsz2:max_work_block_size", max_work_block_size);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(a fixed rate compressor which uses block exponents + rounding)");
    set(options, "frsz2:bits", "the number of bits to use per symbol");
    set(options, "frsz2:max_exp_block_size", "the number of elements to have a common exponent");
    set(options, "frsz2:max_work_block_size", "the number of elements to processes together");
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "frsz2:bits", &bits);
    get(options, "frsz2:max_exp_block_size", &max_exp_block_size);
    get(options, "frsz2:max_work_block_size", &max_work_block_size);
    return 0;
  }

#define dispatch_case(func, bits)                                                                            \
  case bits:                                                                                                 \
    return func<bits, T, ExpType>(data, total_elements, compressed)

  int dispatch_float_compress(float* data, uint64_t total_elements, uint8_t* compressed)
  {
    using T=float;
    using ExpType = int16_t;
    switch (bits) {
      dispatch_case(compress_impl_typed, 8);
      dispatch_case(compress_impl_typed, 9);
      dispatch_case(compress_impl_typed, 10);
      dispatch_case(compress_impl_typed, 11);
      dispatch_case(compress_impl_typed, 12);
      dispatch_case(compress_impl_typed, 13);
      dispatch_case(compress_impl_typed, 14);
      dispatch_case(compress_impl_typed, 15);
      dispatch_case(compress_impl_typed, 16);
      dispatch_case(compress_impl_typed, 17);
      dispatch_case(compress_impl_typed, 18);
      dispatch_case(compress_impl_typed, 19);
      dispatch_case(compress_impl_typed, 20);
      dispatch_case(compress_impl_typed, 21);
      dispatch_case(compress_impl_typed, 22);
      dispatch_case(compress_impl_typed, 23);
      dispatch_case(compress_impl_typed, 24);
      dispatch_case(compress_impl_typed, 25);
      dispatch_case(compress_impl_typed, 26);
      dispatch_case(compress_impl_typed, 27);
      dispatch_case(compress_impl_typed, 28);
      dispatch_case(compress_impl_typed, 29);
      dispatch_case(compress_impl_typed, 30);
      dispatch_case(compress_impl_typed, 31);
      dispatch_case(compress_impl_typed, 32);
      default:
        throw std::runtime_error("unsupported number of bits: " + std::to_string(bits));
    }
  }
  int dispatch_double_compress(double* data, uint64_t total_elements, uint8_t* compressed)
  {
    using T=double;
    using ExpType = int16_t;
    switch (bits) {
      dispatch_case(compress_impl_typed, 8);
      dispatch_case(compress_impl_typed, 9);
      dispatch_case(compress_impl_typed, 10);
      dispatch_case(compress_impl_typed, 11);
      dispatch_case(compress_impl_typed, 12);
      dispatch_case(compress_impl_typed, 13);
      dispatch_case(compress_impl_typed, 14);
      dispatch_case(compress_impl_typed, 15);
      dispatch_case(compress_impl_typed, 16);
      dispatch_case(compress_impl_typed, 17);
      dispatch_case(compress_impl_typed, 18);
      dispatch_case(compress_impl_typed, 19);
      dispatch_case(compress_impl_typed, 20);
      dispatch_case(compress_impl_typed, 21);
      dispatch_case(compress_impl_typed, 22);
      dispatch_case(compress_impl_typed, 23);
      dispatch_case(compress_impl_typed, 24);
      dispatch_case(compress_impl_typed, 25);
      dispatch_case(compress_impl_typed, 26);
      dispatch_case(compress_impl_typed, 27);
      dispatch_case(compress_impl_typed, 28);
      dispatch_case(compress_impl_typed, 29);
      dispatch_case(compress_impl_typed, 30);
      dispatch_case(compress_impl_typed, 31);
      dispatch_case(compress_impl_typed, 32);
      dispatch_case(compress_impl_typed, 33);
      dispatch_case(compress_impl_typed, 34);
      dispatch_case(compress_impl_typed, 35);
      dispatch_case(compress_impl_typed, 36);
      dispatch_case(compress_impl_typed, 37);
      dispatch_case(compress_impl_typed, 38);
      dispatch_case(compress_impl_typed, 39);
      dispatch_case(compress_impl_typed, 40);
      dispatch_case(compress_impl_typed, 41);
      dispatch_case(compress_impl_typed, 42);
      dispatch_case(compress_impl_typed, 43);
      dispatch_case(compress_impl_typed, 44);
      dispatch_case(compress_impl_typed, 45);
      dispatch_case(compress_impl_typed, 46);
      dispatch_case(compress_impl_typed, 47);
      dispatch_case(compress_impl_typed, 48);
      dispatch_case(compress_impl_typed, 49);
      dispatch_case(compress_impl_typed, 50);
      dispatch_case(compress_impl_typed, 51);
      dispatch_case(compress_impl_typed, 52);
      dispatch_case(compress_impl_typed, 53);
      dispatch_case(compress_impl_typed, 54);
      dispatch_case(compress_impl_typed, 55);
      dispatch_case(compress_impl_typed, 56);
      dispatch_case(compress_impl_typed, 57);
      dispatch_case(compress_impl_typed, 58);
      dispatch_case(compress_impl_typed, 59);
      dispatch_case(compress_impl_typed, 60);
      dispatch_case(compress_impl_typed, 61);
      dispatch_case(compress_impl_typed, 62);
      dispatch_case(compress_impl_typed, 63);
      dispatch_case(compress_impl_typed, 64);
      default:
        throw std::runtime_error("unsupported number of bits: " + std::to_string(bits));
    }
  }
  int dispatch_float_decompress(float* data, uint64_t total_elements, uint8_t* compressed)
  {
    using T = float;
    using ExpType = int16_t;
    switch (bits) {
      dispatch_case(decompress_impl_typed, 8);
      dispatch_case(decompress_impl_typed, 9);
      dispatch_case(decompress_impl_typed, 10);
      dispatch_case(decompress_impl_typed, 11);
      dispatch_case(decompress_impl_typed, 12);
      dispatch_case(decompress_impl_typed, 13);
      dispatch_case(decompress_impl_typed, 14);
      dispatch_case(decompress_impl_typed, 15);
      dispatch_case(decompress_impl_typed, 16);
      dispatch_case(decompress_impl_typed, 17);
      dispatch_case(decompress_impl_typed, 18);
      dispatch_case(decompress_impl_typed, 19);
      dispatch_case(decompress_impl_typed, 20);
      dispatch_case(decompress_impl_typed, 21);
      dispatch_case(decompress_impl_typed, 22);
      dispatch_case(decompress_impl_typed, 23);
      dispatch_case(decompress_impl_typed, 24);
      dispatch_case(decompress_impl_typed, 25);
      dispatch_case(decompress_impl_typed, 26);
      dispatch_case(decompress_impl_typed, 27);
      dispatch_case(decompress_impl_typed, 28);
      dispatch_case(decompress_impl_typed, 29);
      dispatch_case(decompress_impl_typed, 30);
      dispatch_case(decompress_impl_typed, 31);
      dispatch_case(decompress_impl_typed, 32);
      default:
        throw std::runtime_error("unsupported number of bits: " + std::to_string(bits));
    }
  }
  int dispatch_double_decompress(double* data, uint64_t total_elements, uint8_t* compressed)
  {
    using T = double;
    using ExpType = int16_t;
    switch (bits) {
      dispatch_case(decompress_impl_typed, 8);
      dispatch_case(decompress_impl_typed, 9);
      dispatch_case(decompress_impl_typed, 10);
      dispatch_case(decompress_impl_typed, 11);
      dispatch_case(decompress_impl_typed, 12);
      dispatch_case(decompress_impl_typed, 13);
      dispatch_case(decompress_impl_typed, 14);
      dispatch_case(decompress_impl_typed, 15);
      dispatch_case(decompress_impl_typed, 16);
      dispatch_case(decompress_impl_typed, 17);
      dispatch_case(decompress_impl_typed, 18);
      dispatch_case(decompress_impl_typed, 19);
      dispatch_case(decompress_impl_typed, 20);
      dispatch_case(decompress_impl_typed, 21);
      dispatch_case(decompress_impl_typed, 22);
      dispatch_case(decompress_impl_typed, 23);
      dispatch_case(decompress_impl_typed, 24);
      dispatch_case(decompress_impl_typed, 25);
      dispatch_case(decompress_impl_typed, 26);
      dispatch_case(decompress_impl_typed, 27);
      dispatch_case(decompress_impl_typed, 28);
      dispatch_case(decompress_impl_typed, 29);
      dispatch_case(decompress_impl_typed, 30);
      dispatch_case(decompress_impl_typed, 31);
      dispatch_case(decompress_impl_typed, 32);
      dispatch_case(decompress_impl_typed, 33);
      dispatch_case(decompress_impl_typed, 34);
      dispatch_case(decompress_impl_typed, 35);
      dispatch_case(decompress_impl_typed, 36);
      dispatch_case(decompress_impl_typed, 37);
      dispatch_case(decompress_impl_typed, 38);
      dispatch_case(decompress_impl_typed, 39);
      dispatch_case(decompress_impl_typed, 40);
      dispatch_case(decompress_impl_typed, 41);
      dispatch_case(decompress_impl_typed, 42);
      dispatch_case(decompress_impl_typed, 43);
      dispatch_case(decompress_impl_typed, 44);
      dispatch_case(decompress_impl_typed, 45);
      dispatch_case(decompress_impl_typed, 46);
      dispatch_case(decompress_impl_typed, 47);
      dispatch_case(decompress_impl_typed, 48);
      dispatch_case(decompress_impl_typed, 49);
      dispatch_case(decompress_impl_typed, 50);
      dispatch_case(decompress_impl_typed, 51);
      dispatch_case(decompress_impl_typed, 52);
      dispatch_case(decompress_impl_typed, 53);
      dispatch_case(decompress_impl_typed, 54);
      dispatch_case(decompress_impl_typed, 55);
      dispatch_case(decompress_impl_typed, 56);
      dispatch_case(decompress_impl_typed, 57);
      dispatch_case(decompress_impl_typed, 58);
      dispatch_case(decompress_impl_typed, 59);
      dispatch_case(decompress_impl_typed, 60);
      dispatch_case(decompress_impl_typed, 61);
      dispatch_case(decompress_impl_typed, 62);
      dispatch_case(decompress_impl_typed, 63);
      dispatch_case(decompress_impl_typed, 64);
      default:
        throw std::runtime_error("unsupported number of bits: " + std::to_string(bits));
    }
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

  template<uint8_t bits, class T, class ExpType>
  int decompress_impl_typed(T* output, const uint64_t total_elements, uint8_t const* compressed)
  {
    runtime_assert("the work blocks must be byte aligned", (max_work_block_size * bits) % 8 == 0);
    runtime_assert("the exp block must be as large or larger than the work_block",
                   max_work_block_size <= max_exp_block_size);
    using InputType = storage_t<bits>;
    using OutputType = scaled_t<T>;

    const size_t num_exp_blocks = ceildiv(total_elements, max_exp_block_size);
    const size_t exp_block_bytes = ceildiv(max_exp_block_size * bits, 8) + sizeof(ExpType);
    for (size_t exp_block_id = 0; exp_block_id < num_exp_blocks; exp_block_id++) {

      const size_t exp_block_elements =
        std::min(max_exp_block_size, total_elements - exp_block_id * max_exp_block_size);
      const size_t num_work_blocks = ceildiv(exp_block_elements, max_work_block_size);
      const size_t work_block_bytes = ceildiv(max_work_block_size * bits, 8);
      const size_t exp_block_data_offset = max_exp_block_size * exp_block_id;
      const uint8_t* exp_block_compressed = compressed + exp_block_bytes * exp_block_id;

      // recover the exponent
      ExpType block_exp = {};
      memcpy(&block_exp, exp_block_compressed, sizeof(ExpType));

      // recover the scaled values
      std::vector<scaled_t<T>> exp_block_scaled(max_exp_block_size);
      for (size_t work_block_id = 0; work_block_id < num_work_blocks; ++work_block_id) {
        const size_t work_block_elements =
          std::min(max_work_block_size, exp_block_elements - work_block_id * max_work_block_size);
        size_t output_bit_offset = 0;
        size_t output_byte_offset = sizeof(ExpType) + work_block_id * work_block_bytes;
        for (size_t i = 0; i < work_block_elements; ++i) {
          InputType tmp[2] = { 0, 0 };
          const uint16_t copy_size =
            (bits + output_bit_offset > 8 * sizeof(InputType)) ? 2 * sizeof(InputType) : sizeof(InputType);
          const uint16_t safe_copy_size = std::min<uint16_t>(exp_block_bytes-output_byte_offset, copy_size);
          memcpy(tmp, exp_block_compressed + output_byte_offset, safe_copy_size);
          exp_block_scaled[i + work_block_id * max_work_block_size] =
            shift_left<OutputType>(tmp, bits, output_bit_offset);

          output_byte_offset += (output_bit_offset + bits) / 8;
          output_bit_offset = (output_bit_offset + bits) % 8;
        }
      }

      // de-scale the values
      for (size_t i = 0; i < exp_block_elements; ++i) {
        output[i + exp_block_data_offset] = fp::fixed_to_floating<T>(exp_block_scaled[i], block_exp);
      }
    }

    return 0;
  }

  template<uint8_t bits, class T, class ExpType>
  int compress_impl_typed(T const* data, const uint64_t total_elements, uint8_t* compressed)
  {
    runtime_assert("the work blocks must be byte aligned", (max_work_block_size * bits) % 8 == 0);
    runtime_assert("the exp block must be as large or larger than the work_block",
                   max_work_block_size <= max_exp_block_size);

    using InputType = scaled_t<T>;
    using OutputType = storage_t<bits>;

    const size_t num_exp_blocks = ceildiv(total_elements, max_exp_block_size);
    const size_t exp_block_bytes = ceildiv(max_exp_block_size * bits, 8) + sizeof(ExpType);
    for (size_t exp_block_id = 0; exp_block_id < num_exp_blocks; exp_block_id++) {

      // how many elements to process in this block?
      const size_t exp_block_elements =
        std::min(max_exp_block_size, total_elements - exp_block_id * max_exp_block_size);

      // find the max exponent in the block to determine the bias
      const size_t exp_block_data_offset = max_exp_block_size * exp_block_id;
      T in_max = 0;
      for (size_t i = 0; i < exp_block_elements; ++i) {
        in_max = std::max(in_max, std::fabs(data[i + exp_block_data_offset]));
      }
      ExpType max_exp = fp::exponent(in_max);


      // preform the scaling
      std::vector<scaled_t<T>> exp_block_scaled(max_exp_block_size);
      for (size_t i = 0; i < exp_block_elements; ++i) {
        exp_block_scaled[i] = fp::floating_to_fixed(data[i + exp_block_data_offset], max_exp);
      }

      // compute the exp_block offset
      uint8_t* exp_block_compressed = compressed + exp_block_bytes * exp_block_id;
      bzero(exp_block_compressed, exp_block_bytes);
      memcpy(exp_block_compressed, &max_exp, sizeof(ExpType));

      // at this point we have scaled values that we can encode

      const size_t num_work_blocks = ceildiv(exp_block_elements, max_work_block_size);
      const size_t work_block_bytes = ceildiv(max_work_block_size * bits, 8);

      for (size_t work_block_id = 0; work_block_id < num_work_blocks; ++work_block_id) {
        const size_t work_block_elements =
          std::min(max_work_block_size, exp_block_elements - work_block_id * max_work_block_size);
        size_t output_bit_offset = 0;
        size_t output_byte_offset = sizeof(ExpType) + work_block_id * work_block_bytes;
        for (size_t i = 0; i < work_block_elements; ++i) {
          const scaled_t<OutputType> to_store =
            exp_block_scaled[i + work_block_id * max_work_block_size] >> (sizeof(InputType) * 8 - bits);

          // we do this dance to avoid an unaligned load
          uint8_t* const out = exp_block_compressed + output_byte_offset;
          uint8_t const copy_size =
            ((bits + output_bit_offset > 8 * sizeof(OutputType)) ? 2 * sizeof(OutputType) : sizeof(OutputType));
          uint8_t const safe_copy_size = std::min<uint8_t>(copy_size, exp_block_bytes-output_byte_offset);
          OutputType temp[2] = { 0, 0 };

          memcpy(temp, out, safe_copy_size);

          // write everything from offset to output_type boundary
          uint8_t first_output_shift;
          uint8_t first_output_bits;
          if (bits + output_bit_offset > 8 * sizeof(OutputType)) {
            first_output_shift = 0;
            first_output_bits = 8 * sizeof(OutputType) - output_bit_offset;
          } else {
            first_output_shift = 8 * sizeof(OutputType) - (bits + output_bit_offset);
            first_output_bits = bits;
          }

          const OutputType first_store =
            (to_store >> (bits - first_output_bits) &
             (ones_t<OutputType>::value >> (8 * sizeof(OutputType) - first_output_bits)))
            << first_output_shift;
          temp[0] |= first_store;

          // if there are leftovers, write those to the high-order bytes
          if (bits + output_bit_offset > 8 * sizeof(OutputType)) {
            const uint8_t remaining = bits - first_output_bits;
            const uint8_t second_shift = 8 * sizeof(OutputType) - remaining;
            const OutputType second_store =
              (to_store & (ones_t<OutputType>::value >> (8 * sizeof(OutputType) - remaining)))
              << second_shift;
            temp[1] |= second_store;
          }

          // copy it back to avoid an unaligned store
          memcpy(out, temp, safe_copy_size);

          output_byte_offset += (bits + output_bit_offset) / 8;
          output_bit_offset = (output_bit_offset + bits) % 8;
        }
      }
    }

    return 0;
  }

  size_t output_size(size_t total_elements) const
  {
    using ExpType = int16_t;
    const size_t num_exp_blocks = ceildiv(total_elements, max_exp_block_size);
    return (ceildiv(max_exp_block_size * bits, 8) + sizeof(ExpType)) * num_exp_blocks;
  }

  int compress_impl(const pressio_data* input, struct pressio_data* output) override
  {
    try {
      *output = pressio_data::owning(pressio_byte_dtype, { output_size(input->num_elements()) });

      if (input->dtype() == pressio_float_dtype) {
        return dispatch_float_compress(
          static_cast<float*>(input->data()), input->num_elements(), static_cast<uint8_t*>(output->data()));
      } else if (input->dtype() == pressio_double_dtype) {
        return dispatch_double_compress(
          static_cast<double*>(input->data()), input->num_elements(), static_cast<uint8_t*>(output->data()));
      } else {
        return set_error(1, "unsupported dtype");
      }
    } catch (std::exception const& ex) {
      return set_error(2, ex.what());
    }
  }

  int decompress_impl(const pressio_data* input, struct pressio_data* output) override
  {
    try {
      if (output->dtype() == pressio_float_dtype) {
        return dispatch_float_decompress(
          static_cast<float*>(output->data()), output->num_elements(), static_cast<uint8_t*>(input->data()));
      } else if (output->dtype() == pressio_double_dtype) {
        return dispatch_double_decompress(
          static_cast<double*>(output->data()), output->num_elements(), static_cast<uint8_t*>(input->data()));
      } else {
        return set_error(1, "unsupported dtype");
      }
    } catch (std::exception const& ex) {
      return set_error(2, ex.what());
    }
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "frsz2"; }

  pressio_options get_metrics_results_impl() const override { return {}; }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<frsz2_compressor_plugin>(*this);
  }

  uint64_t bits = 32;
  uint64_t max_exp_block_size = 1024;
  uint64_t max_work_block_size = 8;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "frsz2", []() {
  return compat::make_unique<frsz2_compressor_plugin>();
});

}
}
