//
// Created by root on 12/18/24.
//

#include <cute/tensor.hpp>

void test_tuple() {
    auto A = cute::make_tuple(int{1}, cute::Int<2>{});
    auto B = cute::make_tuple(cute::Int<3>{}, cute::Int<4>{});
    auto C = cute::make_tuple(A, B);
    auto D = cute::make_tuple(cute::Int<5>{}, A, C);
    auto E = cute::make_tuple(D, int{6});

    cute::print(cute::rank(E));
    std::cout << std::endl;
    cute::print(cute::rank<0>(E));
    std::cout << std::endl;
    cute::print(cute::get<0>(E));
    std::cout << std::endl;
    cute::print(cute::depth(E));
    std::cout << std::endl;
    cute::print(cute::size(E));
    std::cout << std::endl;
}

template <class Shape, class Stride>
void print1D(cute::Layout<Shape,Stride> const& layout) {
    for (int i = 0; i < size(layout); ++i) {
        printf("%3d  ", layout(i));
    }
    printf("\n");
}

void test_layout() {
    auto A = cute::make_layout(cute::make_shape(3, 4), cute::make_stride(1, 3));
    cute::print_layout(A);
    print1D(A);
    std::cout << std::endl;
    auto B = cute::make_layout(cute::make_shape(3, 4), cute::make_stride(4, 1));
    cute::print_layout(B);
    print1D(B);

    // idx2crd
    auto C = cute::make_shape(cute::Int<5>{}, cute::make_shape(cute::Int<4>{}, 3));
    cute::print(cute::idx2crd(8, C));
    std::cout << std::endl;

    // crd2idx
    auto D = cute::make_layout(cute::make_shape(cute::Int<5>{}, cute::make_shape(cute::Int<4>{}, 3)), cute::LayoutLeft{});
    cute::print(cute::crd2idx(cute::make_coord(3, cute::make_coord(2, 1)), D.shape<>(), D.shape<>()));
    std::cout << std::endl;

    // coalesce
    auto E = cute::Layout<cute::Shape <cute::_2,cute::Shape <cute::_1,cute::_6>>, cute::Stride<cute::_1,cute::Stride<cute::_6,cute::_2>>>{};
    auto coalesceE = cute::coalesce(E);
    cute::print(coalesceE);
    std::cout << std::endl;

    // composition tiler
    auto F = cute::make_layout(cute::make_shape(12, cute::make_shape(4, 8)), cute::make_stride(59, cute::make_stride(13, 1)));
    auto tilerF = cute::make_tile(cute::Layout<cute::_3, cute::_4>{}, cute::Layout<cute::_8, cute::_1>{});
    auto compositionF = cute::composition(F, tilerF);
    cute::print(F);
    std::cout << std::endl;
    cute::print(tilerF);
    std::cout << std::endl;
    cute::print(compositionF);
    std::cout << std::endl;

    // logical divide
    auto G = cute::make_layout(cute::make_shape(cute::Int<9>{}, cute::make_shape(cute::Int<4>{}, cute::Int<8>{})),
                               cute::make_stride(cute::Int<59>{}, cute::make_stride(cute::Int<13>{}, cute::Int<1>{})));
    auto tilerG = cute::make_tile(cute::Layout<cute::_3, cute::_3>{},
                                 cute::Layout<cute::Shape<cute::_2, cute::_4>, cute::Stride<cute::_1, cute::_8>>{});
    auto logicalDivideG = cute::logical_divide(G, tilerG);
    cute::print(logicalDivideG);
    std::cout << std::endl;

    // zipped divide
    auto zippedDivideG = cute::zipped_divide(G, tilerG);
    cute::print(zippedDivideG);
    std::cout << std::endl;
}


int main(int argc, char **argv) {
    test_tuple();
    test_layout();
    return 0;
}