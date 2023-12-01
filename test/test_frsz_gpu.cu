#include "gtest/gtest.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <type_traits>
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
      // Use unary operator+ to convert the value into an integer or float to print properly
      std::cerr << i << ": " << +v1[i] << " vs. " << +v2[i] << '\n';
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
  const bool i_to_f_devices = compare_kernel<1, num_elems>(bit_cast_from_to<IType, FType>{}, i_mem, f_mem);
  EXPECT_TRUE(i_to_f_devices);
  const auto cmp_f = compare_vectors(f_mem.get_host_vector(), expected_f);
  EXPECT_TRUE(cmp_f);
  const auto f_to_i_devices = compare_kernel<1, num_elems>(bit_cast_from_to<FType, IType>{}, f_mem, i_mem);
  EXPECT_TRUE(f_to_i_devices);
  const auto cmp_i = compare_vectors(i_mem.get_host_vector(), expected_i);
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
  // Note: bool vector needs to be copied explicitly because underlying type is not a `std::vector`
  auto cmp_sign = compare_vectors(sign_mem.get_host_copy(), expected_sign);
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
    if (i > 0 && i % 8 == 0) {
      std::cout << '\n' << std::setw(2) << i << ": ";
    }
    std::cout << std::setw(2) << std::setfill('0') << std::hex << static_cast<int>(bytes[i]) << std::dec
              << ' ';
  }
  std::cout << std::dec << '\n';
}

template<int bits_per_value, int max_exp_block_size, class FpType, class ExpType>
void
launch_both_compressions(const std::vector<FpType>& flt_vec)
{
  using frsz2_comp = frsz::frsz2_compressor<bits_per_value, max_exp_block_size, FpType, ExpType>;
  // std::cout << "Test with " << int(bits_per_value) << "bits; Exponent block size: " << max_exp_block_size
  //           << '\n';
  Memory<FpType> flt_mem(flt_vec);
  const std::size_t total_elements = flt_mem.get_num_elems();
  constexpr int blocks_per_tb = std::max(1, 512 / max_exp_block_size);
  // constexpr int blocks_per_tb = 1;
  constexpr int comp_num_threads = 4 * max_exp_block_size;
  const int comp_num_blocks = frsz::ceildiv<int>(total_elements, comp_num_threads);
  const int decomp_num_threads = blocks_per_tb * max_exp_block_size;
  const int decomp_num_blocks = frsz::ceildiv<int>(total_elements, decomp_num_threads);
  const std::size_t compressed_memory_size = frsz2_comp::compute_compressed_memory_size_byte(total_elements);

  Memory<std::uint8_t> compressed_mem(compressed_memory_size, 0xFF);
  frsz2_comp::compress_cpu_impl(flt_mem.get_host_const(), total_elements, compressed_mem.get_host());

  frsz::compress_gpu<frsz2_comp, comp_num_threads><<<comp_num_blocks, comp_num_threads>>>(
    flt_mem.get_device_const(), total_elements, compressed_mem.get_device());
  // std::cout << "Device compressed memory:\n";
  // print_bytes(compressed_mem.get_device_copy().data(), compressed_memory_size);
  // std::cout << "Host compressed memory:\n";
  // print_bytes(compressed_mem.get_host(), compressed_memory_size);

  EXPECT_TRUE(compressed_mem.is_device_matching_host());
  const auto device_compressed = compressed_mem.get_device_copy();
  bool is_compressed_same = true;
  for (std::size_t i = 0; i < compressed_memory_size; ++i) {
    const auto hval = compressed_mem.get_host_vector()[i];
    const auto dval = device_compressed[i];
    if (hval != dval) {
      // Use unary operator+ to promote it to integer type
      std::cerr << i << ": host " << +hval << " vs " << +dval << " device\n";
      is_compressed_same = false;
    }
  }
  EXPECT_TRUE(is_compressed_same);
  // EXPECT_TRUE(compressed_mem.is_device_matching_host());
  // compressed_mem.to_device();
  // compressed_mem.to_host();
  // if ((bits_per_value == 15 || bits_per_value == 16) && max_exp_block_size == 8) {
  //   print_bytes(compressed_mem.get_host(), compressed_memory_size);
  // }

  flt_mem.set_all_to(std::numeric_limits<FpType>::infinity());

  frsz::decompress_gpu<frsz2_comp><<<decomp_num_blocks, decomp_num_threads>>>(
    flt_mem.get_device(), total_elements, compressed_mem.get_device_const());
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

TEST(frsz2_gpu, decompress)
{
  using f_type = double;
  std::array<double, 9> repeat_vals{ 1., 2., 3., 4., 0.25, -0.25, -0.125, 1 / 32., 0.125 };
  const std::size_t total_size{ 2049 };
  // const std::size_t total_size{ 49 };
  // const std::size_t total_size{ 9 };
  std::vector<f_type> vect(total_size);
  for (std::size_t i = 0; i < total_size; ++i) {
    vect[i] = repeat_vals[i % repeat_vals.size()];
  }
  launch_both_compressions<32, 32, f_type, std::int16_t>(vect);
  launch_both_compressions<16, 32, f_type, std::int16_t>(vect);
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

template<typename FpType>
std::enable_if_t<std::is_floating_point<FpType>::value, void>
convert_back_and_forth(const std::vector<FpType>& input_values)
{
  // FIXME adopt the exponent type to something smaller after they are independent
  using exponent_type =
    std::conditional_t<std::is_same<FpType, double>::value,
                       std::int64_t,
                       std::conditional_t<std::is_same<FpType, float>::value, std::int32_t, std::int16_t>>;
  constexpr int max_exp_block_size = 1; // Use just one to make sure conversions back and forth work
  constexpr int bits_per_value = sizeof(FpType) * CHAR_BIT;

  launch_both_compressions<bits_per_value, max_exp_block_size, FpType, exponent_type>(input_values);
}

TEST(frsz2_gpu, back_and_forth32)
{

  // TODO brute-force all floating point numbers (incl. denormals)
  using fp_type = float;
  constexpr int num_vectors = 100000;
  std::random_device rd;
  std::default_random_engine rnd_engine(rd());
  std::uniform_real_distribution<fp_type> uni_dist(-1e8, 1e8);
  std::vector<fp_type> rnd_vector(num_vectors);
  for (auto&& el : rnd_vector) {
    auto rnd_val = uni_dist(rnd_engine);
    while (!std::isfinite(rnd_val)) {
      std::cout << "Re-randomizing non-finite number " << rnd_val << '\n';
      rnd_val = uni_dist(rnd_engine);
    }
    el = rnd_val;
  }
  convert_back_and_forth(rnd_vector);
}
