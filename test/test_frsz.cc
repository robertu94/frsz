#include "gtest/gtest.h"
#include <bitset>
#include <cmath>
#include <frsz.h>
#include <iomanip>
#include <iostream>
#include <libpressio_ext/cpp/libpressio.h>
#include <limits>
#include <vector>
#include <bit>

TEST(frsz, integation1) {
  register_frsz();

  std::vector<float> f{10.0, 20.0, 30.0, 20.0, 10.0};
  pressio_data in =
      pressio_data::copy(pressio_float_dtype, f.data(), {f.size()});
  pressio_data out = pressio_data::owning(pressio_float_dtype, {f.size()});
  pressio_data compressed =
      pressio_data::owning(pressio_float_dtype, {f.size()});

  pressio library;
  pressio_compressor c = library.get_compressor("frsz");
  c->set_options({{"frsz:epsilon", 5.0}});
  c->compress(&in, &compressed);
  c->decompress(&compressed, &out);

  auto decomp_f = out.to_vector<float>();
  EXPECT_EQ(f, decomp_f);
}

template <class T>
using scaled_t = std::conditional_t<
    sizeof(T) == 8, uint64_t,
    std::conditional_t<sizeof(T) == 4, uint32_t,
                       std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>>;
template <size_t N>
using storage_t = std::conditional_t<
    (N <= 8), uint8_t,
    std::conditional_t<(N <= 16), uint16_t,
                       std::conditional_t<(N <= 32), uint32_t, uint64_t>>>;

template <class T>
inline constinit const int expbits =
    static_cast<int>(log2(std::numeric_limits<T>::max_exponent)) + 1; template <class T>
inline constinit const int ebias = ((1 << (expbits<T> - 1)) - 1);

template <class T>
std::pair<std::vector<scaled_t<T>>, int> scale_block(T const *in_begin,
                                                     T const *in_end) {
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
  return {scaled, e};
}

template <class T>
std::vector<T> restore_block(scaled_t<T> const *begin, scaled_t<T> const *end,
                             int expmax) {
  size_t N = std::distance(begin, end);
  std::vector<T> out(N);
  T scale_factor =
      ldexp(1, expmax - (static_cast<int>(CHAR_BIT * sizeof(T)) - 2));
  for (size_t i = 0; i < N; ++i) {
    out[i] = begin[i] * scale_factor;
  }

  return out;
}

constexpr int ceildiv(int a, int b) {
  if (a % b == 0)
    return a / b;
  else
    return a / b + 1;
}


template <class T>
struct ones_t;

template <>
struct ones_t<uint8_t> {
    static constexpr uint8_t value = 0xFF;
};
template <>
struct ones_t<uint16_t> {
    static constexpr uint16_t value = 0xFFFF;
};
template <>
struct ones_t<uint32_t> {
    static constexpr uint32_t value = 0xFFFFFFFF;
};
template <>
struct ones_t<uint64_t> {
    static constexpr uint64_t  value = 0xFFFFFFFFFFFFFFFF;
};
template <>
struct ones_t<int8_t> {
    static constexpr uint8_t value = 0xFF;
};
template <>
struct ones_t<int16_t> {
    static constexpr uint16_t value = 0xFFFF;
};
template <>
struct ones_t<int32_t> {
    static constexpr uint32_t value = 0xFFFFFFFF;
};
template <>
struct ones_t<int64_t> {
    static constexpr uint64_t value = 0xFFFFFFFFFFFFFFFF;
};

/**
 * \param[in] in pointer to the byte containing the bytes to shift
 * \param[in] bits how many bits to shift left
 * \param[in] offset what bit does the number start on?
 */
template <class OutputType, class InputType>
OutputType shift_left(InputType const* in, uint8_t bits, uint8_t offset) {
    assert(offset < 8);
    assert(bits <= sizeof(InputType)*8);
    OutputType result = in[0];
    result <<= offset;
    if(bits + offset > sizeof(InputType)*8) {
        const uint8_t remaining = bits-(sizeof(InputType)*8-offset);
        OutputType remaining_bits = ((in[1]>>(sizeof(InputType)*8-remaining))&(ones_t<InputType>::value>>remaining));
        remaining_bits <<= (sizeof(InputType)*8-bits);
        result |= remaining_bits;
    }
    if(sizeof(OutputType)>sizeof(InputType)) {
        result <<= 8*(sizeof(OutputType)-sizeof(InputType));
    }
    return result;
}


TEST(frsz2, shift_left) {
    {
        uint32_t input[] = {0x00000000, 0xFFFFFFFF};
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 0), 0x00000000);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 1), 0x00000001);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 2), 0x00000003);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 3), 0x00000007);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 4), 0x0000000F);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 5), 0x0000001F);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 6), 0x0000003F);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 7), 0x0000007F);
        EXPECT_EQ(shift_left<uint16_t>(input, 16, 0), 0x0000);
        EXPECT_EQ(shift_left<uint16_t>(input, 16, 1), 0x0000);
        EXPECT_EQ(shift_left<uint16_t>(input, 16, 2), 0x0000);
        EXPECT_EQ(shift_left<uint16_t>(input, 16, 3), 0x0000);
        EXPECT_EQ(shift_left<uint16_t>(input, 16, 4), 0x0000);
        EXPECT_EQ(shift_left<uint16_t>(input, 16, 5), 0x0000);
        EXPECT_EQ(shift_left<uint16_t>(input, 16, 6), 0x0000);
        EXPECT_EQ(shift_left<uint16_t>(input, 16, 7), 0x0000);
        EXPECT_EQ(shift_left<uint8_t>(input, 8, 0), 0x00);
        EXPECT_EQ(shift_left<uint8_t>(input, 8, 1), 0x00);
        EXPECT_EQ(shift_left<uint8_t>(input, 8, 2), 0x00);
        EXPECT_EQ(shift_left<uint8_t>(input, 8, 3), 0x00);
        EXPECT_EQ(shift_left<uint8_t>(input, 8, 4), 0x00);
        EXPECT_EQ(shift_left<uint8_t>(input, 8, 5), 0x00);
        EXPECT_EQ(shift_left<uint8_t>(input, 8, 6), 0x00);
        EXPECT_EQ(shift_left<uint8_t>(input, 8, 7), 0x00);
    }
    {
        uint32_t input[] = {0xFFFFFFFF, 0x00000000};
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 0), 0xFFFFFFFF);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 1), 0xFFFFFFFE);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 2), 0xFFFFFFFC);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 3), 0xFFFFFFF8);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 4), 0xFFFFFFF0);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 5), 0xFFFFFFE0);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 6), 0xFFFFFFC0);
        EXPECT_EQ(shift_left<uint32_t>(input, 32, 7), 0xFFFFFF80);
    }
}

template <class OutputType>
struct expected_offset_t {
  size_t byte;
  size_t bit;
  OutputType to_store;
  OutputType first_store;
  size_t first_output_shift;
  size_t first_output_bits;
  std::optional<OutputType> second_store;
  std::optional<size_t> second_shift;
};

template <uint8_t bits, class InputType, class OutputType>
std::vector<uint8_t> test_compress(
        std::vector<InputType> const& scaled,
        int8_t const e,
        std::vector<expected_offset_t<OutputType>> const& expected_offsets
) {
  unsigned int output_bit_offset = 0;
  unsigned int output_byte_offset = sizeof(int8_t);
  std::vector<uint8_t> result (ceildiv(scaled.size()*bits + sizeof(int8_t)*8,8));
  result[0] = std::bit_cast<uint8_t>(static_cast<int8_t>(e));
  for (size_t i = 0; i < scaled.size(); ++i) {
     scaled_t<OutputType> to_store = scaled[i] >> (sizeof(InputType)*8 - bits);

     //we do this dance to avoid an unaligned load
     uint8_t* out = result.data() + output_byte_offset;
     uint8_t copy_size = (bits + output_bit_offset > 8*sizeof(OutputType)) ? 2*sizeof(OutputType) : sizeof(OutputType);
     OutputType temp[2] = {0, 0};
     memcpy(temp, out, copy_size);

     EXPECT_EQ(expected_offsets[i].byte, output_byte_offset) << "bits " << static_cast<int>(bits) << " index " << i;
     EXPECT_EQ(expected_offsets[i].bit, output_bit_offset) << "bits " << static_cast<int>(bits) << " index " << i;
     EXPECT_EQ(expected_offsets[i].to_store, to_store) << "bits " << static_cast<int>(bits) << " index " << i;
     //write everything from offset to output_type boundary
     uint8_t first_output_shift;
     uint8_t first_output_bits;
     if(bits+output_bit_offset > 8*sizeof(OutputType)) {
         first_output_shift = 0;
         first_output_bits = 8*sizeof(OutputType)-output_bit_offset;
     } else {
         first_output_shift = 8*sizeof(OutputType) - (bits + output_bit_offset);
         first_output_bits = bits;
     }
     EXPECT_EQ(expected_offsets[i].first_output_bits, first_output_bits) << "bits " << static_cast<int>(bits) << " index " << i;
     EXPECT_EQ(expected_offsets[i].first_output_shift, first_output_shift) << "bits " << static_cast<int>(bits) << " index " << i;

     OutputType first_store = (to_store>>(bits-first_output_bits) & 
             (ones_t<OutputType>::value >> (8*sizeof(OutputType) - first_output_bits))) << first_output_shift;
     EXPECT_EQ(expected_offsets[i].first_store, first_store) << "bits " << static_cast<int>(bits) << " index " << i;
     temp[0] |= first_store; //TODO
                             
     
     //if there are leftovers, write those to the high-order bytes
     if(bits + output_bit_offset > 8*sizeof(OutputType)) {
           uint8_t remaining = bits - first_output_bits;
           uint8_t second_shift = 8*sizeof(OutputType) - remaining;
           OutputType second_store = (to_store & (ones_t<OutputType>::value >> (8*sizeof(OutputType) - remaining) )) << second_shift;
           temp[1] |= second_store;
           EXPECT_EQ(expected_offsets[i].second_store, second_store) << "bits " << static_cast<int>(bits) << " index " << i;
           EXPECT_EQ(expected_offsets[i].second_shift, second_shift) << "bits " << static_cast<int>(bits) << " index " << i;
     } else {
         EXPECT_EQ(expected_offsets[i].second_store, std::optional<size_t>{}) << "bits " << static_cast<int>(bits) << " index " << i;
         EXPECT_EQ(expected_offsets[i].second_shift, std::optional<size_t>{}) << "bits " << static_cast<int>(bits) << " index " << i;
     }

     //copy it back
     memcpy(out, temp, copy_size);

     output_byte_offset += (bits+output_bit_offset)/ 8;
     output_bit_offset = (output_bit_offset + bits) % 8; }

  return result;
}

template <class OutputType>
struct expected_offsets_decompress_t{
    OutputType loaded;
};

template <uint8_t bits, class OutputType>
std::vector<OutputType> test_decompress(std::vector<uint8_t> const& input, size_t N,
        std::vector<expected_offsets_decompress_t<OutputType>> const& expected_offsets) {
    using InputType = storage_t<bits>;
    size_t bit_offset = 0;
    size_t byte_offset = sizeof(int8_t);
    std::vector<OutputType> ret(N);

    for(size_t i = 0; i < N; ++i) {
        InputType tmp[2] = {0, 0};
        uint16_t copy_size = (bits + bit_offset > 8*sizeof(InputType)) ? 2*sizeof(InputType) : sizeof(InputType);
        memcpy(tmp, input.data()+byte_offset, copy_size);
        ret[i] = shift_left<OutputType>(tmp, bits, bit_offset);
        EXPECT_EQ(ret[i], expected_offsets[i].loaded) << "decompress bits " << static_cast<int>(bits) << " offset " << i;

        byte_offset += (bit_offset + bits) / 8;
        bit_offset = (bit_offset + bits) % 8;
    }

    return ret;
}

TEST(frsz2, prototype) {
  std::array f{1.0f, 2.0f, 2.33f, 4.0f, 8.0f, 16.0f, 32.0f};
  auto const& [scaled, e] = scale_block(f.data(), f.data() + f.size());
  auto restored =
      restore_block<float>(scaled.data(), scaled.data() + scaled.size(), e);
  for (size_t i = 0; i < f.size(); ++i) {
      EXPECT_LT(std::abs(f[i]-restored[i]), .01);
  }
  static_assert(sizeof(storage_t<32>) == 4, "test");

  {
      constexpr size_t bits = 16;
      std::vector<expected_offset_t<storage_t<bits>>> expected_offsets {
          {1,0,  0x0100, 0x0100, 0, 16, {}, {}},
          {3,0,  0x0200, 0x0200, 0, 16, {}, {}},
          {5,0,  0x0254, 0x0254, 0, 16, {}, {}},
          {7,0,  0x0400, 0x0400, 0, 16, {}, {}},
          {9,0,  0x0800, 0x0800, 0, 16, {}, {}},
          {11,0, 0x1000, 0x1000, 0, 16, {}, {}},
          {13,0, 0x2000, 0x2000, 0, 16, {}, {}},
      };
      auto result = test_compress<bits>(scaled, static_cast<int8_t>(e), expected_offsets);
      std::vector<expected_offsets_decompress_t<scaled_t<float>>> expected_decompress {
          {0x01000000},
          {0x02000000},
          {0x02540000},
          {0x04000000},
          {0x08000000},
          {0x10000000},
          {0x20000000}
      };
      test_decompress<bits>(result, scaled.size(), expected_decompress);
  }

  {
      constexpr size_t bits = 8;
      std::vector<expected_offset_t<storage_t<bits>>> expected_offsets {
          {1,0,  0x01, 0x01, 0,  8, {}, {}},
          {2,0,  0x02, 0x02, 0,  8, {}, {}},
          {3,0,  0x02, 0x02, 0,  8, {}, {}},
          {4,0,  0x04, 0x04, 0,  8, {}, {}},
          {5,0,  0x08, 0x08, 0,  8, {}, {}},
          {6,0, 0x10, 0x10, 0,   8, {}, {}},
          {7,0, 0x20, 0x20, 0,   8, {}, {}},
      };
      std::vector<expected_offsets_decompress_t<scaled_t<float>>> expected_decompress {
          {0x01000000},
          {0x02000000},
          {0x02000000},
          {0x04000000},
          {0x08000000},
          {0x10000000},
          {0x20000000}
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
