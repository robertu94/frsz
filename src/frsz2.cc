#include "frsz2.hpp"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace libpressio {
namespace frsz2_ns {

constexpr int
ceildiv(int a, int b)
{
  if (a % b == 0)
    return a / b;
  else
    return a / b + 1;
}

class frsz2_compressor_plugin : public libpressio_compressor_plugin
{
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "frsz2:bits", bits);
    set(options, "frsz2:max_exp_block_size", max_exp_block_size);
    // set(options, "frsz2:max_work_block_size", max_work_block_size);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    
        std::vector<std::string> invalidations {"frsz2:bits", "frsz2:max_exp_block_size"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(a fixed rate compressor which uses block exponents + rounding)");
    set(options, "frsz2:bits", "the number of bits to use per symbol");
    set(options,
        "frsz2:max_exp_block_size",
        "the number of elements to have a common exponent. Supported values: 2, 4, 8, 16, 32, 64, 128, 256, "
        "512, 1024");
    // set(options, "frsz2:max_work_block_size", "the number of elements to processes together");
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "frsz2:bits", &bits);
    get(options, "frsz2:max_exp_block_size", &max_exp_block_size);
    // get(options, "frsz2:max_work_block_size", &max_work_block_size);
    return 0;
  }

  // TODO Remove
  template<uint8_t bits, class T, class ExpType>
  int decompress_impl_typed(T*, const uint64_t, uint8_t const*)
  {
    return 0;
  }
  // TODO Remove
  template<uint8_t bits, class T, class ExpType>
  int compress_impl_typed(const T*, const uint64_t, uint8_t*)
  {
    return 0;
  }

#define dispatch_case(func, bits)                                                                            \
  case bits:                                                                                                 \
    return func<bits, T>(data, total_elements, compressed)

  int dispatch_float_compress(float* data, uint64_t total_elements, uint8_t* compressed)
  {
    using T = float;
    using exp_list = frsz::int_list_t<2, 4, 8, 16, 32, 64, 128, 256, 512, 1024>;
    // clang-format off
    using bit_list = frsz::int_list_t<4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                      21,22,23,24,25,26,27,28,29,30,31,32>;
    // clang-format on
    frsz::dispatch_frsz2_compression<T>(
      bit_list{}, bits, exp_list{}, max_exp_block_size, data, total_elements, compressed);
    return 0;
  }
  int dispatch_double_compress(double* data, uint64_t total_elements, uint8_t* compressed)
  {
    using T = double;
    using exp_list = frsz::int_list_t<2, 4, 8, 16, 32, 64, 128, 256, 512, 1024>;
    // clang-format off
    using bit_list = frsz::int_list_t<4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                      21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,
                                      36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
                                      51,52,53,54,55,56,57,58,59,60,61,62,63,64>;
    // clang-format on
    frsz::dispatch_frsz2_compression<T>(
      bit_list{}, bits, exp_list{}, max_exp_block_size, data, total_elements, compressed);
    return 0;
  }
  int dispatch_float_decompress(float* data, uint64_t total_elements, uint8_t* compressed)
  {
    using T = float;
    using exp_list = frsz::int_list_t<2, 4, 8, 16, 32, 64, 128, 256, 512, 1024>;
    // clang-format off
    using bit_list = frsz::int_list_t<4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                      21,22,23,24,25,26,27,28,29,30,31,32>;
    // clang-format on
    frsz::dispatch_frsz2_decompression<T>(
      bit_list{}, bits, exp_list{}, max_exp_block_size, data, total_elements, compressed);
    return 0;
  }
  int dispatch_double_decompress(double* data, uint64_t total_elements, uint8_t* compressed)
  {
    using T = double;
    using exp_list = frsz::int_list_t<2, 4, 8, 16, 32, 64, 128, 256, 512, 1024>;
    // clang-format off
    using bit_list = frsz::int_list_t<4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                                      21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,
                                      36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,
                                      51,52,53,54,55,56,57,58,59,60,61,62,63,64>;
    // clang-format on
    frsz::dispatch_frsz2_decompression<T>(
      bit_list{}, bits, exp_list{}, max_exp_block_size, data, total_elements, compressed);
    return 0;
  }

  size_t output_size(size_t total_elements) const
  {
    // TODO replace with call to the frsz2_compressor
    const std::size_t uint_compressed_size_bit = frsz::detail::get_next_power_of_two_value(bits);
    const std::size_t uint_compressed_size_byte = uint_compressed_size_bit / CHAR_BIT;
    const std::size_t exponent_size_byte = uint_compressed_size_byte;
    const std::size_t compressed_block_size_byte =
      ceildiv(max_exp_block_size * bits, uint_compressed_size_bit) * uint_compressed_size_byte +
      exponent_size_byte;

    const std::size_t remainder = total_elements % max_exp_block_size;
    return (total_elements / max_exp_block_size) * compressed_block_size_byte +
           (remainder > 0) * (exponent_size_byte + ceildiv(remainder * bits, uint_compressed_size_bit) *
                                                     uint_compressed_size_byte);
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
  int patch_version() const override { return 2; }
  const char* version() const override { return "0.0.2"; }
  const char* prefix() const override { return "frsz2"; }

  pressio_options get_metrics_results_impl() const override { return {}; }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<frsz2_compressor_plugin>(*this);
  }

  uint64_t bits = 32;
  uint64_t max_exp_block_size = 32;
  // uint64_t max_work_block_size = 8;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "frsz2", []() {
  return compat::make_unique<frsz2_compressor_plugin>();
});

}
}
