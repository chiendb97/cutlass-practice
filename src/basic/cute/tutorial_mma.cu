//
// Created by root on 12/24/24.
//

#include "cute/tensor.hpp"

void test_trait() {
    auto mma = cute::MMA_Atom<cute::SM70_8x8x4_F16F16F16F16_NT>{};
    cute::print_latex(mma);
}

int main(int argc, char **argv) {
    test_trait();
    return 0;
}