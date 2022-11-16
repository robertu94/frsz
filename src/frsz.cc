#include <frsz_version.h>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include <cmath>
#include <algorithm>
#include <limits>

extern "C" void register_frsz() {

}

namespace libpressio { namespace frsz_ns {

class frsz_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  { struct pressio_options options;
    set(options, "frsz:epsilon", epsilon);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(a fixed rate version of the SZ Algorithm)");
    set(options, "frsz:epsilon", "\"absolute error bound\" applied during quantization; because frsz does not store unpredictable values, this is not a true absolute error bound");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "frsz:epsilon", &epsilon);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {

    switch(input->dtype()) {
      case pressio_float_dtype:
        return compress_1d<float>(input, output);
      case pressio_double_dtype:
        return compress_1d<double>(input, output);
      default:
        return set_error(1, "unsupported type: currently only floating point supported");
    }

    return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    switch(output->dtype()) {
      case pressio_float_dtype:
        return decompress_1d<float>(input, output);
        break;
      case pressio_double_dtype:
        return decompress_1d<double>(input, output);
        break;
      default:
        return set_error(1, "unsupported type: currently only floating point supported");

    }

    return 0;
  }

  int major_version() const override { return  FRSZ_MAJOR_VERSION; }
  int minor_version() const override { return  FRSZ_MINOR_VERSION; }
  int patch_version() const override { return FRSZ_PATCH_VERSION; }
  const char* version() const override { return FRSZ_VERSION; }
  const char* prefix() const override { return "frsz"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<frsz_compressor_plugin>(*this);
  }

private:

  template<class Out, class In> static Out saturate(In i) noexcept {
    if (i < std::numeric_limits<Out>::lowest()) return std::numeric_limits<Out>::lowest();
    else if (i > std::numeric_limits<Out>::max()) return std::numeric_limits<Out>::max();
    else return static_cast<Out>(i);
  }

  template <class T, size_t BLOCK=256>
  int compress_1d(pressio_data const* input, pressio_data* output) {
    auto quant_mem = pressio_data::owning(input->dtype(), {input->num_elements()});
    const size_t len = input->normalized_dims(1).at(0);
    const size_t output_size_in_bytes = len*sizeof(uint16_t);
    if(output->capacity_in_bytes() < output_size_in_bytes) {
      *output = pressio_data::owning(pressio_byte_dtype, {output_size_in_bytes});
    } else {
      output->set_dimensions({output_size_in_bytes});
      output->set_dtype(pressio_byte_dtype);
    }
    T const* data = static_cast<T*>(input->data());
    T* quantized = static_cast<T*>(quant_mem.data());
    int16_t* compressed = static_cast<int16_t*>(output->data());
    const double ebx2_inv = 1.0/(epsilon * 2.0);

#pragma omp parallel
    {
#pragma omp for
      for (size_t i = 0; i < len; ++i) {
        quantized[i] = std::round(data[i] * ebx2_inv);
      }
#pragma omp barrier
#pragma omp for
      for (size_t i = 0; i < len; ++i) {
        T delta = quantized[i] - ((i%BLOCK == 0) ? 0 : quantized[i-1]);
        compressed[i] = saturate<int16_t>(delta);
      }
    }
    return 0;
  }

  template <class T, size_t BLOCK=256>
  int decompress_1d(pressio_data const* input, pressio_data* output) {
    const size_t len = output->normalized_dims(1).at(0);
    int16_t* quantized = static_cast<int16_t*>(input->data());
    T* output_data = static_cast<T*>(output->data());
    const double ebx2 = epsilon * 2;

#pragma omp parallel for
    for (size_t block_idx = 0; block_idx < (len/BLOCK)+1; ++block_idx) {
      if(BLOCK*block_idx < len){
        output_data[BLOCK*block_idx] = static_cast<T>(quantized[BLOCK*block_idx]) * ebx2;
        for (size_t i = 1; i < BLOCK; ++i) {
          auto global_idx = BLOCK*block_idx+i;
          if(global_idx < len) {
            output_data[global_idx] = (static_cast<T>(quantized[i])*ebx2) + output_data[global_idx-1] ;
          }
        }
      }
    }

    return 0;
  }

  double epsilon = 1e-4;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "frsz", []() {
  return compat::make_unique<frsz_compressor_plugin>();
});

} }
