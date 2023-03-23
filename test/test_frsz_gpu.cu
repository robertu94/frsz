#include "gtest/gtest.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "frsz2.hpp"
#include "memory.cuh"

template<int NumBlocks, int NumThreads, class Functor, class... Args>
__global__ void
test_kernel(Functor func, Args... args)
{
  if (threadIdx.x < NumThreads && blockIdx.x < NumBlocks) {
    func(blockIdx.x * blockDim.x + threadIdx.x, args...);
  }
}

template<class From, class To>
struct bit_cast_from_to
{
public:
  void __device__ __host__ operator()(int idx, const From* from, To* to) const
  {
    to[idx] = frsz::xstd::bit_cast<To>(from[idx]);
  }
};

template<int NumBlocks, int NumThreads, class Functor, class... Args>
bool
compare_kernel(Functor func, Args&&... args)
{
  for (int idx = 0; idx < NumBlocks * NumThreads; ++idx) {
    func(idx, args.get_host()...);
  }
  test_kernel<NumBlocks, NumThreads><<<NumBlocks, NumThreads>>>(func, args.get_device()...);
  return compare_gpu_cpu(std::forward<Args>(args)...);
}

template<class T>
bool
compare_vectors(const std::vector<T>& v1, const std::vector<T>& v2)
{
  if (v1.size() != v2.size()) {
    return false;
  }
  bool are_equal = true;
  for (std::size_t i = 0; i < v1.size(); ++i) {
    if (v1[i] != v2[i]) {
      are_equal = false;
      std::cerr << i << ": " << v1[i] << " vs. " << v2[i] << '\n';
    }
  }
  return are_equal;
}

template<int num_elems, class IType, class FType>
void
perform_bit_cast_tests(const std::vector<IType>& expected_i, const std::vector<FType>& expected_f)
{
  static_assert(sizeof(IType) == sizeof(FType), "Sizes of types have to match!");
  ASSERT_EQ(expected_i.size(), expected_f.size());
  ASSERT_EQ(expected_i.size(), num_elems);
  Memory<IType> i_mem(expected_i);
  Memory<FType> f_mem(expected_f.size(), {});
  bool i_to_f_devices = compare_kernel<1, num_elems>(bit_cast_from_to<IType, FType>{}, i_mem, f_mem);
  EXPECT_TRUE(i_to_f_devices);
  auto cmp_f = compare_vectors(f_mem.get_host_vector(), expected_f);
  EXPECT_TRUE(cmp_f);
  auto f_to_i_devices = compare_kernel<1, num_elems>(bit_cast_from_to<FType, IType>{}, f_mem, i_mem);
  EXPECT_TRUE(f_to_i_devices);
  auto cmp_i = compare_vectors(i_mem.get_host_vector(), expected_i);
  EXPECT_TRUE(cmp_i);
}

TEST(frsz2_gpu, TestBitCast32)
{
  using f_type = float;
  using i_type = std::int32_t;
  using ui_type = std::uint32_t;
  std::vector<ui_type> expected_ui({ 0b0'01111111'01000000000000000000000,
                                     0b0'00000000'00000000000000000000001,
                                     0b0'00000001'00000000000000000000000,
                                     0b1'11111111'00000000000000000000000,
                                     0b0'00000000'00000000000000000000000,
                                     0b1'01111111'00000000000000000000001,
                                     0b0'01101000'00000000000000000000000,
                                     0b0'10000011'01010000000000000000000 });
  std::vector<i_type> expected_i(expected_ui.size(), {});
  for (std::size_t i = 0; i < expected_i.size(); ++i) {
    expected_i[i] = static_cast<i_type>(expected_ui[i]);
    // Make sure the bit-pattern did not change as long as numbers are stored in
    // the 2's complement
    // unsigned -> signed is implementation defined for numbers > max(signed),
    // but signed -> unsigned keeps the bit pattern if 2's complement is used
    assert(static_cast<ui_type>(expected_i[i]) == expected_ui[i]);
  }
  const std::vector<f_type> expected_f{ 1.25f,
                                        std::numeric_limits<f_type>::denorm_min(),
                                        std::numeric_limits<f_type>::min(),
                                        -std::numeric_limits<f_type>::infinity(),
                                        0.f,
                                        -1.f - std::numeric_limits<f_type>::epsilon(),
                                        std::numeric_limits<f_type>::epsilon(),
                                        21.f };
  std::cout << std::scientific << std::setprecision(8);
  std::cerr << std::scientific << std::setprecision(8);
  perform_bit_cast_tests<8>(expected_ui, expected_f);
  perform_bit_cast_tests<8>(expected_i, expected_f);
}

TEST(frsz2_gpu, TestBitCast64)
{
  using f_type = double;
  using i_type = std::int64_t;
  using ui_type = std::uint64_t;
  std::vector<ui_type> expected_ui({ 0b0'01111111111'0100000000000000000000000000000000000000000000000000,
                                     0b0'00000000000'0000000000000000000000000000000000000000000000000001,
                                     0b0'00000000001'0000000000000000000000000000000000000000000000000000,
                                     0b1'11111111111'0000000000000000000000000000000000000000000000000000,
                                     0b0'00000000000'0000000000000000000000000000000000000000000000000000,
                                     0b1'01111111111'0000000000000000000000000000000000000000000000000001,
                                     0b0'01111001011'0000000000000000000000000000000000000000000000000000,
                                     0b0'10000000011'0101000000000000000000000000000000000000000000000000 });
  std::vector<i_type> expected_i(expected_ui.size(), {});
  for (std::size_t i = 0; i < expected_i.size(); ++i) {
    expected_i[i] = static_cast<i_type>(expected_ui[i]);
    // Make sure the bit-pattern did not change as long as numbers are stored in
    // the 2's complement
    // unsigned -> signed is implementation defined for numbers > max(signed),
    // but signed -> unsigned keeps the bit pattern if 2's complement is used
    assert(static_cast<ui_type>(expected_i[i]) == expected_ui[i]);
  }
  const std::vector<f_type> expected_f{ 1.25,
                                        std::numeric_limits<f_type>::denorm_min(),
                                        std::numeric_limits<f_type>::min(),
                                        -std::numeric_limits<f_type>::infinity(),
                                        0.,
                                        -1. - std::numeric_limits<f_type>::epsilon(),
                                        std::numeric_limits<f_type>::epsilon(),
                                        21. };
  std::cout << std::scientific << std::setprecision(17);
  std::cerr << std::scientific << std::setprecision(17);
  perform_bit_cast_tests<8>(expected_ui, expected_f);
  perform_bit_cast_tests<8>(expected_i, expected_f);
}

template<class FloatingType>
struct dismantle_floating_value
{
private:
  using sign_t = typename frsz::detail::fp::float_traits<FloatingType>::sign_t;
  using exponent_t = typename frsz::detail::fp::float_traits<FloatingType>::exponent_t;
  using significand_t = typename frsz::detail::fp::float_traits<FloatingType>::significand_t;

public:
  void __device__ __host__ operator()(int idx,
                                      const FloatingType* src,
                                      sign_t* sign,
                                      exponent_t* exponent,
                                      significand_t* significand) const
  {
    const auto val = src[idx];
    sign[idx] = frsz::detail::fp::sign(val);
    exponent[idx] = frsz::detail::fp::exponent(val);
    significand[idx] = frsz::detail::fp::significand(val);
  }
};

template<int num_elems, class FType>
void
perform_dismantle_floating_value(
  const std::vector<FType>& input,
  const std::vector<typename frsz::detail::fp::float_traits<FType>::sign_t>& expected_sign,
  const std::vector<typename frsz::detail::fp::float_traits<FType>::exponent_t>& expected_exponent,
  const std::vector<typename frsz::detail::fp::float_traits<FType>::significand_t>& expected_significand)
{
  using sign_t = typename frsz::detail::fp::float_traits<FType>::sign_t;
  using exponent_t = typename frsz::detail::fp::float_traits<FType>::exponent_t;
  using significand_t = typename frsz::detail::fp::float_traits<FType>::significand_t;
  ASSERT_EQ(input.size(), num_elems);
  ASSERT_EQ(input.size(), expected_sign.size());
  ASSERT_EQ(input.size(), expected_exponent.size());
  ASSERT_EQ(input.size(), expected_significand.size());
  Memory<FType> input_mem(input);
  Memory<sign_t> sign_mem(input.size(), {});
  Memory<exponent_t> exponent_mem(input.size(), {});
  Memory<significand_t> significand_mem(input.size(), {});
  bool cmp_devices = compare_kernel<1, num_elems>(
    dismantle_floating_value<FType>{}, input_mem, sign_mem, exponent_mem, significand_mem);
  EXPECT_TRUE(cmp_devices);
  auto cmp_sign = compare_vectors(sign_mem.get_host_vector(), expected_sign);
  EXPECT_TRUE(cmp_sign);
  auto cmp_exponent = compare_vectors(exponent_mem.get_host_vector(), expected_exponent);
  EXPECT_TRUE(cmp_exponent);
  auto cmp_significand = compare_vectors(significand_mem.get_host_vector(), expected_significand);
  EXPECT_TRUE(cmp_significand);
}

TEST(frsz2_gpu, dismantling_floating)
{
  std::vector<float> input_val32{ 32.f, 0.5f, 0.f, -0.375f };
  std::vector<bool> sign32{ false, false, false, true };
  std::vector<std::int8_t> exponent32{ 5, -1, -127, -2 };
  std::vector<std::uint32_t> significand32{ 1 << 23, 1 << 23, 0, 3 << 22 };

  std::vector<double> input_val64{ -32., 0.5, 0., -0.375 };
  std::vector<bool> sign64{ true, false, false, true };
  std::vector<std::int16_t> exponent64{ 5, -1, -1023, -2 };
  std::vector<std::uint64_t> significand64{
    std::uint64_t{ 1 } << 52, std::uint64_t{ 1 } << 52, 0, std::uint64_t{ 3 } << 51
  };

  perform_dismantle_floating_value<4, float>(input_val32, sign32, exponent32, significand32);
  perform_dismantle_floating_value<4, double>(input_val64, sign64, exponent64, significand64);
}

template<class FloatingType>
struct floating_and_fixed_conversions
{
public:
  using uint_type = frsz::detail::scaled_t<FloatingType>;

  void __device__ __host__ operator()(int idx,
                                      const FloatingType* src,
                                      const std::int16_t* exponent,
                                      uint_type* intermediate,
                                      FloatingType* result) const
  {
    const auto val = src[idx];
    const auto exp = exponent[idx];
    intermediate[idx] = frsz::detail::fp::floating_to_fixed(val, exp);
    result[idx] = frsz::detail::fp::fixed_to_floating<FloatingType>(intermediate[idx], exp);
  }
};

template<int num_elems, class FType>
void
perform_floating_and_fixed_conversions(
  const std::vector<FType>& input,
  const std::vector<std::int16_t>& exponent,
  const std::vector<frsz::detail::scaled_t<FType>>& expected_intermediate,
  const std::vector<FType>& expected_output)
{
  ASSERT_EQ(input.size(), num_elems);
  ASSERT_EQ(input.size(), exponent.size());
  ASSERT_EQ(input.size(), expected_intermediate.size());
  ASSERT_EQ(input.size(), expected_output.size());
  Memory<FType> input_mem(input);
  Memory<std::int16_t> exponent_mem(exponent);
  Memory<frsz::detail::scaled_t<FType>> intermediate_mem(expected_intermediate.size(), {});
  Memory<FType> output_mem(input.size(), {});
  bool cmp_devices = compare_kernel<1, num_elems>(
    floating_and_fixed_conversions<FType>{}, input_mem, exponent_mem, intermediate_mem, output_mem);
  EXPECT_TRUE(cmp_devices);
  auto cmp_intermediate = compare_vectors(intermediate_mem.get_host_vector(), expected_intermediate);
  EXPECT_TRUE(cmp_intermediate);
  auto cmp_output = compare_vectors(output_mem.get_host_vector(), expected_output);
  EXPECT_TRUE(cmp_output);
}

// TODO brute-force all fp32 values for fixed_to_floating and floating_to_fixed
TEST(frsz2_gpu, floating_to_fixed)
{
  std::vector<float> input32{
    32.f, 2.f, frsz::xstd::bit_cast<float>(0b0'00000000'11111111111111111111111), 0.125f, -0.375f
  };
  std::vector<std::int16_t> exp32{ 5, 5, 5, 2, 1 };
  std::vector<std::uint32_t> intermediate32{ std::uint32_t{ 0x800000 } << 7,
                                             std::uint32_t{ 0x800000 } << 7 - 4,
                                             0,
                                             std::uint32_t{ 0x02000000 },
                                             std::uint32_t{ 0x8C000000 } };
  std::vector<float> output32{
    32.f, 2.f, frsz::xstd::bit_cast<float>(0b0'01100101'00000000000000000000000), 0.125f, -0.375f
  };

  std::vector<double> input64{ 32., 2., -0.125, 0.375 };
  std::vector<std::int16_t> exp64{ 5, 5, 2, -1 };
  std::vector<std::uint64_t> intermediate64{ std::uint64_t{ 1 } << 52 + 10,
                                             std::uint64_t{ 1 } << 52 + 10 - 4,
                                             std::uint64_t{ 0x82 } << 56,
                                             std::uint64_t{ 3 } << 60 };
  std::vector<double> output64{ 32., 2., -0.125, 0.375 };

  perform_floating_and_fixed_conversions<5>(input32, exp32, intermediate32, output32);
  perform_floating_and_fixed_conversions<4>(input64, exp64, intermediate64, output64);
}

void
print_bytes(const std::uint8_t* bytes, std::size_t size)
{
  for (std::size_t i = 0; i < size; ++i) {
    if (i % 8 == 0) {
      std::cout << '\n' << std::setw(2) << i << ": ";
    }
    std::cout << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(bytes[i]) << std::dec
              << ' ';
  }
  std::cout << std::dec << '\n';
}

template<std::uint8_t bits_per_value, int max_exp_block_size, class FpType, class ExpType>
void
launch_both_compressions(const std::vector<FpType>& flt_vec)
{
  using frsz2_comp = frsz::frsz2_compressor<bits_per_value, max_exp_block_size, FpType, ExpType>;
  // std::cout << "Test with " << int(bits_per_value) << "bits; Exponent block size: " << max_exp_block_size
  //           << '\n';
  Memory<FpType> flt_mem(flt_vec);
  const std::size_t total_elements = flt_mem.get_num_elems();
  const int num_threads = max_exp_block_size;
  const int num_blocks = frsz::ceildiv<int>(total_elements, num_threads);
  const std::size_t compressed_memory_size = frsz2_comp::compute_compressed_memory_size_byte(total_elements);

  Memory<std::uint8_t> compressed_mem(compressed_memory_size, 0xFF);
  frsz2_comp::compress_cpu_impl(flt_mem.get_host_const(), total_elements, compressed_mem.get_host());

  frsz::compress_gpu<frsz2_comp>
    <<<num_blocks, num_threads>>>(flt_mem.get_device_const(), total_elements, compressed_mem.get_device());
  // std::cout << "Device compressed memory:\n";
  // print_bytes(compressed_mem.get_device_copy().data(), compressed_memory_size);
  // std::cout << "Host compressed memory:\n";
  // print_bytes(compressed_mem.get_host(), compressed_memory_size);

  // EXPECT_TRUE(compressed_mem.is_device_matching_host());
  const auto device_compressed = compressed_mem.get_device_copy();
  bool is_compressed_same = true;
  for (std::size_t i = 0; i < compressed_memory_size; ++i) {
    const auto hval = int(compressed_mem.get_host_vector()[i]);
    const auto dval = int(device_compressed[i]);
    if (hval != dval) {
      // std::cerr << i << ": host " << hval << " vs " << dval << " device\n";
      is_compressed_same = false;
    }
  }
  EXPECT_TRUE(is_compressed_same);
  // EXPECT_TRUE(compressed_mem.is_device_matching_host());
  // compressed_mem.to_device();
  // compressed_mem.to_host();

  std::vector<FpType> overwrite_memory_flt(total_elements, std::numeric_limits<FpType>::infinity());
  flt_mem.set_memory_to(overwrite_memory_flt);

  frsz::decompress_gpu<frsz2_comp>
    <<<num_blocks, num_threads>>>(flt_mem.get_device(), total_elements, compressed_mem.get_device_const());
  cudaDeviceSynchronize();
  frsz2_comp::decompress_cpu_impl(flt_mem.get_host(), total_elements, compressed_mem.get_host_const());
  // flt_mem.print_device_host();
  EXPECT_TRUE(flt_mem.is_device_matching_host());
  const auto no_loss = compare_vectors(flt_mem.get_device_copy(), flt_vec);
  EXPECT_TRUE(no_loss);
  // for (std::size_t i = 0; i < total_elements; ++i) {
  //   std::cout << std::setw(2) << i << ": " << flt_vec[i] << " -> " << flt_mem.get_host()[i] << '\n';
  // }
}

// TODO
TEST(frsz2_gpu, decompress)
{
  using f_type = double;
  std::array<double, 9> repeat_vals{ 1., 2., 3., 4., 0.25, -0.25, -0.125, 1 / 32., 0.125 };
  const std::size_t total_size{ 2049 };
  std::vector<f_type> vect(total_size);
  for (std::size_t i = 0; i < total_size; ++i) {
    vect[i] = repeat_vals[i % repeat_vals.size()];
  }
  launch_both_compressions<16, 8, f_type, std::int16_t>(vect);
  launch_both_compressions<15, 8, f_type, std::int16_t>(vect);
  launch_both_compressions<9, 8, f_type, std::int16_t>(vect);
  launch_both_compressions<9, 4, f_type, std::int16_t>(vect);
  launch_both_compressions<9, 5, f_type, std::int16_t>(vect);

  using f_type2 = float;
  std::vector<f_type2> vect2(total_size + 111);
  for (std::size_t i = 0; i < vect2.size(); ++i) {
    vect2[i] = repeat_vals[i % repeat_vals.size()];
  }
  launch_both_compressions<16, 8, f_type2, std::int16_t>(vect2);
  launch_both_compressions<15, 8, f_type2, std::int8_t>(vect2);
  launch_both_compressions<9, 4, f_type2, std::int8_t>(vect2);
  launch_both_compressions<9, 5, f_type2, std::int8_t>(vect2);
  // launch_both_compressions<4, 8, f_type2, std::int8_t>(vect2);
}

////////////////////////////////////////////////////////////////////////////////

#if false

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
constexpr const int expbits = static_cast<int>(
  8 * sizeof(uint32_t) - std::countl_zero(static_cast<uint32_t>(std::numeric_limits<T>::max_exponent)));
template<class T>
constinit const int ebias = ((1 << (expbits<T> - 1)) - 1);

template<class T>
std::pair<std::vector<scaled_t<T>>, int>
scale_block(T const* in_begin, T const* in_end)
{
  size_t N = std::distance(in_begin, in_end);

  T in_max = 0;
  for (size_t i = 0; i < N; ++i) {
    in_max = std::max(in_max, std::fabs(in_begin[i]));
  }

  int e = -ebias<T>;
  if (in_max >= std::numeric_limits<T>::min()) {
    frexp(in_max, &e);
  }
  T scale_factor = ldexp(1, (static_cast<int>(CHAR_BIT * sizeof(T)) - 2) - e);

  std::vector<scaled_t<T>> scaled(N);
  for (size_t i = 0; i < N; ++i) {
    scaled[i] = static_cast<scaled_t<T>>(scale_factor * in_begin[i]);
  }
  return { scaled, e };
}

template<class T>
std::vector<T>
restore_block(scaled_t<T> const* begin, scaled_t<T> const* end, int expmax)
{
  size_t N = std::distance(begin, end);
  std::vector<T> out(N);
  T scale_factor = ldexp(1, expmax - (static_cast<int>(CHAR_BIT * sizeof(T)) - 2));
  for (size_t i = 0; i < N; ++i) {
    out[i] = begin[i] * scale_factor;
  }

  return out;
}

template<class OutputType>
struct expected_offset_t
{
  size_t byte;
  size_t bit;
  OutputType to_store;
  OutputType first_store;
  size_t first_output_shift;
  size_t first_output_bits;
  std::optional<OutputType> second_store;
  std::optional<size_t> second_shift;
};

template<uint8_t bits, class InputType, class OutputType>
std::vector<uint8_t>
test_compress(std::vector<InputType> const& scaled,
              int8_t const e,
              std::vector<expected_offset_t<OutputType>> const& expected_offsets)
{
  unsigned int output_bit_offset = 0;
  unsigned int output_byte_offset = sizeof(int8_t);
  std::vector<uint8_t> result(ceildiv(scaled.size() * bits + sizeof(int8_t) * 8, 8));
  result[0] = std::bit_cast<uint8_t>(static_cast<int8_t>(e));
  for (size_t i = 0; i < scaled.size(); ++i) {
    scaled_t<OutputType> to_store = scaled[i] >> (sizeof(InputType) * 8 - bits);

    // we do this dance to avoid an unaligned load
    uint8_t* out = result.data() + output_byte_offset;
    uint8_t copy_size =
      (bits + output_bit_offset > 8 * sizeof(OutputType)) ? 2 * sizeof(OutputType) : sizeof(OutputType);
    OutputType temp[2] = { 0, 0 };
    memcpy(temp, out, copy_size);

    EXPECT_EQ(expected_offsets[i].byte, output_byte_offset)
      << "bits " << static_cast<int>(bits) << " index " << i;
    EXPECT_EQ(expected_offsets[i].bit, output_bit_offset)
      << "bits " << static_cast<int>(bits) << " index " << i;
    EXPECT_EQ(expected_offsets[i].to_store, to_store) << "bits " << static_cast<int>(bits) << " index " << i;
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
    EXPECT_EQ(expected_offsets[i].first_output_bits, first_output_bits)
      << "bits " << static_cast<int>(bits) << " index " << i;
    EXPECT_EQ(expected_offsets[i].first_output_shift, first_output_shift)
      << "bits " << static_cast<int>(bits) << " index " << i;

    OutputType first_store = (to_store >> (bits - first_output_bits) &
                              (ones_t<OutputType>::value >> (8 * sizeof(OutputType) - first_output_bits)))
                             << first_output_shift;
    EXPECT_EQ(expected_offsets[i].first_store, first_store)
      << "bits " << static_cast<int>(bits) << " index " << i;
    temp[0] |= first_store; // TODO

    // if there are leftovers, write those to the high-order bytes
    if (bits + output_bit_offset > 8 * sizeof(OutputType)) {
      uint8_t remaining = bits - first_output_bits;
      uint8_t second_shift = 8 * sizeof(OutputType) - remaining;
      OutputType second_store =
        (to_store & (ones_t<OutputType>::value >> (8 * sizeof(OutputType) - remaining))) << second_shift;
      temp[1] |= second_store;
      EXPECT_EQ(expected_offsets[i].second_store, second_store)
        << "bits " << static_cast<int>(bits) << " index " << i;
      EXPECT_EQ(expected_offsets[i].second_shift, second_shift)
        << "bits " << static_cast<int>(bits) << " index " << i;
    } else {
      EXPECT_EQ(expected_offsets[i].second_store, std::optional<size_t>{})
        << "bits " << static_cast<int>(bits) << " index " << i;
      EXPECT_EQ(expected_offsets[i].second_shift, std::optional<size_t>{})
        << "bits " << static_cast<int>(bits) << " index " << i;
    }

    // copy it back
    memcpy(out, temp, copy_size);

    output_byte_offset += (bits + output_bit_offset) / 8;
    output_bit_offset = (output_bit_offset + bits) % 8;
  }

  return result;
}

template<class OutputType>
struct expected_offsets_decompress_t
{
  OutputType loaded;
};

template<uint8_t bits, class OutputType>
std::vector<OutputType>
test_decompress(std::vector<uint8_t> const& input,
                size_t N,
                std::vector<expected_offsets_decompress_t<OutputType>> const& expected_offsets)
{
  using InputType = storage_t<bits>;
  size_t bit_offset = 0;
  size_t byte_offset = sizeof(int8_t);
  std::vector<OutputType> ret(N);

  for (size_t i = 0; i < N; ++i) {
    InputType tmp[2] = { 0, 0 };
    uint16_t copy_size =
      (bits + bit_offset > 8 * sizeof(InputType)) ? 2 * sizeof(InputType) : sizeof(InputType);
    memcpy(tmp, input.data() + byte_offset, copy_size);
    ret[i] = shift_left<OutputType>(tmp, bits, bit_offset);
    EXPECT_EQ(ret[i], expected_offsets[i].loaded)
      << "decompress bits " << static_cast<int>(bits) << " offset " << i;

    byte_offset += (bit_offset + bits) / 8;
    bit_offset = (bit_offset + bits) % 8;
  }

  return ret;
}

TEST(frsz2, prototype)
{
  std::array f{ 1.0f, 2.0f, 2.33f, 4.0f, 8.0f, 16.0f, 32.0f };
  auto const& [scaled, e] = scale_block(f.data(), f.data() + f.size());
  auto restored = restore_block<float>(scaled.data(), scaled.data() + scaled.size(), e);
  for (size_t i = 0; i < f.size(); ++i) {
    EXPECT_LT(std::abs(f[i] - restored[i]), .01);
  }
  static_assert(sizeof(storage_t<32>) == 4, "test");

  {
    constexpr size_t bits = 16;
    std::vector<expected_offset_t<storage_t<bits>>> expected_offsets{
      { 1, 0, 0x0100, 0x0100, 0, 16, {}, {} },  { 3, 0, 0x0200, 0x0200, 0, 16, {}, {} },
      { 5, 0, 0x0254, 0x0254, 0, 16, {}, {} },  { 7, 0, 0x0400, 0x0400, 0, 16, {}, {} },
      { 9, 0, 0x0800, 0x0800, 0, 16, {}, {} },  { 11, 0, 0x1000, 0x1000, 0, 16, {}, {} },
      { 13, 0, 0x2000, 0x2000, 0, 16, {}, {} },
    };
    auto result = test_compress<bits>(scaled, static_cast<int8_t>(e), expected_offsets);
    std::vector<expected_offsets_decompress_t<scaled_t<float>>> expected_decompress{
      { 0x01000000 }, { 0x02000000 }, { 0x02540000 }, { 0x04000000 },
      { 0x08000000 }, { 0x10000000 }, { 0x20000000 }
    };
    test_decompress<bits>(result, scaled.size(), expected_decompress);
  }

  {
    constexpr size_t bits = 8;
    std::vector<expected_offset_t<storage_t<bits>>> expected_offsets{
      { 1, 0, 0x01, 0x01, 0, 8, {}, {} }, { 2, 0, 0x02, 0x02, 0, 8, {}, {} },
      { 3, 0, 0x02, 0x02, 0, 8, {}, {} }, { 4, 0, 0x04, 0x04, 0, 8, {}, {} },
      { 5, 0, 0x08, 0x08, 0, 8, {}, {} }, { 6, 0, 0x10, 0x10, 0, 8, {}, {} },
      { 7, 0, 0x20, 0x20, 0, 8, {}, {} },
    };
    std::vector<expected_offsets_decompress_t<scaled_t<float>>> expected_decompress{
      { 0x01000000 }, { 0x02000000 }, { 0x02000000 }, { 0x04000000 },
      { 0x08000000 }, { 0x10000000 }, { 0x20000000 }
    };
    auto result = test_compress<bits>(scaled, static_cast<int8_t>(e), expected_offsets);
    test_decompress<bits>(result, scaled.size(), expected_decompress);
  }

  {
    constexpr size_t bits = 5;
    // clang-format off
      std::vector<expected_offset_t<storage_t<bits>>> expected_offsets {
        /* i,   byte, bit, to_store, 1_store, 1_shift, 1_bits, 2_store,    2_shift */
         /*0*/ {1,    0,   0b00000,  0b00000, 3,       5,      {},         {}      },
         /*1*/ {1,    5,   0b00000,  0b00000, 0,       3,      0b00000000, 6      },
         /*2*/ {2,    2,   0b00000,  0b00000, 1,       5,      {},         {}      },
         /*3*/ {2,    7,   0b00000,  0b00000, 0,       1,      0b00000000, 4      },
         /*4*/ {3,    4,   0b00001,  0b00000, 0,       4,      0b10000000, 7      },
         /*5*/ {4,    1,   0b00010,  0b01000, 2,       5,      {},         {}      },
         /*6*/ {4,    6,   0b00100,  0b00000, 0,       2,      0b10000000, 5      },
      };
      std::vector<expected_offsets_decompress_t<scaled_t<float>>> expected_decompress {
          {0b00000000000000000000000000000000},
          {0b00000000000000000000000000000000},
          {0b00000000000000000000000000000000},
          {0b00000000000000000000000000000000},
          {0b00001000000000000000000000000000},
          {0b00010000000000000000000000000000},
          {0b00100000000000000000000000000000}
      };
    // clang-format on
    auto result = test_compress<bits>(scaled, static_cast<int8_t>(e), expected_offsets);
    test_decompress<bits>(result, scaled.size(), expected_decompress);
  }
}

/*
TEST(frsz2, integation1)
{
  register_frsz();

  std::vector<double> f{
    10.0, 20.0, 30.0,  20.0, 11.0, 12.0, 13.0, 14.0,

    40.0, 80.0, 120.0, 80.0, 41.0, 48.0, 42.0, 44.0,
  };
  pressio_data in = pressio_data::copy(pressio_double_dtype, f.data(), { f.size() });
  pressio_data out = pressio_data::owning(pressio_double_dtype, { f.size() });
  pressio_data compressed = pressio_data::owning(pressio_byte_dtype, { f.size() });

  pressio library;
  pressio_compressor c = library.get_compressor("frsz2");
  c->set_options({
    { "frsz2:bits", uint64_t{ 32 } },
    { "frsz2:max_exp_block_size", uint64_t{ 8 } },
    { "frsz2:max_work_block_size", uint64_t{ 4 } },
  });
  c->compress(&in, &compressed);
  c->decompress(&compressed, &out);

  auto f_out = static_cast<float*>(out.data());
  for (size_t i = 0; i < f.size(); ++i) {
      EXPECT_LT(std::fabs(f_out[i] - f[i]), 1e-3);
  }

}
*/

#endif
