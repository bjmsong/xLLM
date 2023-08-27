#include <gtest/gtest.h>
#define ARMA_USE_BLAS

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}