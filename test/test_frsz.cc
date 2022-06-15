#include "gtest/gtest.h"
#include <vector>
#include <libpressio_ext/cpp/libpressio.h>
#include <frsz.h>
#include <iostream>

TEST(frsz, integation1) {
  register_frsz();

  std::vector<float> f{10.0, 20.0, 30.0, 20.0, 10.0};
  pressio_data in = pressio_data::copy(pressio_float_dtype, f.data(), {f.size()});
  pressio_data out = pressio_data::owning(pressio_float_dtype, {f.size()});
  pressio_data compressed = pressio_data::owning(pressio_float_dtype, {f.size()});

  pressio library;
  pressio_compressor c = library.get_compressor("frsz");
  c->set_options({{"frsz:epsilon", 5.0}});
  c->compress(&in, &compressed);
  c->decompress(&compressed, &out);

  auto decomp_f = out.to_vector<float>();
  EXPECT_EQ(f, decomp_f);
}
