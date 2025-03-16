#pragma once

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

using namespace cute;

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const& tensor)
{
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;

    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
    // Tensor out = make_tensor_like<To_type>(out);
    // copy(make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout()), out);
    // return out;
}