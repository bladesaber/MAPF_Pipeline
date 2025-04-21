#include <gtest/gtest.h>
#include "../include/BaseFunctions.h"

class BaseFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test matrix setup
        mat1 = Eigen::MatrixXd(3, 3);
        mat1 << 3, 1, 4,
                1, 5, 9,
                2, 6, 5;
        
        // Expected results for argsort_row
        indices_row = Eigen::MatrixXi(3, 3);
        indices_row << 1, 0, 2,
                       0, 1, 2,
                       0, 2, 1;
        
        asscend_indices_row = Eigen::MatrixXi(3, 3);
        asscend_indices_row << 1, 0, 2,
                               0, 1, 2,
                               0, 2, 1;

        // Expected results for argsort_col
        indices_col = Eigen::MatrixXi(3, 3);
        indices_col << 1, 0, 0,
                       2, 1, 2,
                       0, 2, 1;

        asscend_indices_col = Eigen::MatrixXi(3, 3);
        asscend_indices_col << 2, 0, 0,
                               0, 1, 2,
                               1, 2, 1;

        // Test vectors for get_distance
        loc1 = LocXyz(0, 0, 0);
        loc2 = LocXyz(3, 4, 0);
    }

    Eigen::MatrixXd mat1;
    Eigen::MatrixXi indices_row;
    Eigen::MatrixXi indices_col;
    Eigen::MatrixXi asscend_indices_row;
    Eigen::MatrixXi asscend_indices_col;
    LocXyz loc1, loc2;
};

TEST_F(BaseFunctionsTest, TestGetDistance) {
    double distance = MathSpace::get_distance(loc1, loc2);
    EXPECT_DOUBLE_EQ(distance, 5.0);
}

TEST_F(BaseFunctionsTest, TestArgsortRow) {
    Eigen::MatrixXi result(3, 3);
    MathSpace::argsort_row(mat1, result);
    EXPECT_EQ(result, indices_row);
}

TEST_F(BaseFunctionsTest, TestArgsortCol) {
    Eigen::MatrixXi result(3, 3);
    MathSpace::argsort_col(mat1, result);
    EXPECT_EQ(result, indices_col);
}

TEST_F(BaseFunctionsTest, TestSortRow) {
    Eigen::MatrixXd sorted_mat;
    MathSpace::sort_row(mat1, sorted_mat);
    
    // Check if each row is sorted
    for(int i = 0; i < sorted_mat.rows(); i++) {
        for(int j = 0; j < sorted_mat.cols()-1; j++) {
            EXPECT_LE(sorted_mat(i,j), sorted_mat(i,j+1));
        }
    }
}

TEST_F(BaseFunctionsTest, TestRowAsscendRank) {
    Eigen::MatrixXi indices;
    MathSpace::row_asscend_rank(mat1, indices);
    EXPECT_EQ(indices, asscend_indices_row);
}

TEST_F(BaseFunctionsTest, TestColAsscendRank) {
    Eigen::MatrixXi indices;
    MathSpace::col_asscend_rank(mat1, indices);
    EXPECT_EQ(indices, asscend_indices_col);
}

TEST_F(BaseFunctionsTest, TestMinRow) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> min_vec;
    MathSpace::min_row(mat1, min_vec);
    EXPECT_DOUBLE_EQ(min_vec(0), 1.0);
    EXPECT_DOUBLE_EQ(min_vec(1), 1.0);
    EXPECT_DOUBLE_EQ(min_vec(2), 2.0);
}

TEST_F(BaseFunctionsTest, TestMaxRow) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> max_vec;
    MathSpace::max_row(mat1, max_vec);
    EXPECT_DOUBLE_EQ(max_vec(0), 4.0);
    EXPECT_DOUBLE_EQ(max_vec(1), 9.0);
    EXPECT_DOUBLE_EQ(max_vec(2), 6.0);
}

TEST_F(BaseFunctionsTest, TestMinCol) {
    Eigen::Matrix<double, 1, Eigen::Dynamic> min_vec;
    MathSpace::min_col(mat1, min_vec);
    EXPECT_DOUBLE_EQ(min_vec(0), 1.0);
    EXPECT_DOUBLE_EQ(min_vec(1), 1.0);
    EXPECT_DOUBLE_EQ(min_vec(2), 4.0);
}

TEST_F(BaseFunctionsTest, TestMaxCol) {
    Eigen::Matrix<double, 1, Eigen::Dynamic> max_vec;
    MathSpace::max_col(mat1, max_vec);
    EXPECT_DOUBLE_EQ(max_vec(0), 3.0);
    EXPECT_DOUBLE_EQ(max_vec(1), 6.0);
    EXPECT_DOUBLE_EQ(max_vec(2), 9.0);
}

TEST_F(BaseFunctionsTest, TestSumRow) {
    Eigen::Matrix<double, Eigen::Dynamic, 1> sum_vec;
    MathSpace::sum_row(mat1, sum_vec);
    EXPECT_DOUBLE_EQ(sum_vec(0), 8.0);
    EXPECT_DOUBLE_EQ(sum_vec(1), 15.0);
    EXPECT_DOUBLE_EQ(sum_vec(2), 13.0);
}

TEST_F(BaseFunctionsTest, TestSumCol) {
    Eigen::Matrix<double, 1, Eigen::Dynamic> sum_vec;
    MathSpace::sum_col(mat1, sum_vec);
    EXPECT_DOUBLE_EQ(sum_vec(0), 6.0);
    EXPECT_DOUBLE_EQ(sum_vec(1), 12.0);
    EXPECT_DOUBLE_EQ(sum_vec(2), 18.0); 
}

TEST_F(BaseFunctionsTest, TestComputeVectorAngle) {
    double angle = MathSpace::compute_vector_angle(1, 0, 0, 0, 1, 0);
    EXPECT_DOUBLE_EQ(angle, 0.0); // Perpendicular vectors
    
    angle = MathSpace::compute_vector_angle(1, 0, 0, 1, 0, 0);
    EXPECT_DOUBLE_EQ(angle, 1.0); // Same direction
    
    angle = MathSpace::compute_vector_angle(1, 0, 0, -1, 0, 0);
    EXPECT_DOUBLE_EQ(angle, -1.0); // Opposite direction
}

TEST_F(BaseFunctionsTest, TestMinMaxIndices) {
    EXPECT_EQ(MathSpace::min_row_index(mat1, 0), 1); // First row minimum at column 1
    EXPECT_EQ(MathSpace::min_row_index(mat1, 1), 0); // First row minimum at column 2
    EXPECT_EQ(MathSpace::min_row_index(mat1, 2), 0); // First row minimum at column 2

    EXPECT_EQ(MathSpace::max_row_index(mat1, 0), 2); // First row maximum at column 2
    EXPECT_EQ(MathSpace::max_row_index(mat1, 1), 2); // First row maximum at column 2
    EXPECT_EQ(MathSpace::max_row_index(mat1, 2), 1); // First row maximum at column 2

    EXPECT_EQ(MathSpace::min_col_index(mat1, 0), 1); // First column minimum at row 1
    EXPECT_EQ(MathSpace::min_col_index(mat1, 1), 0); // First column minimum at row 2
    EXPECT_EQ(MathSpace::min_col_index(mat1, 2), 0); // First column minimum at row 2

    EXPECT_EQ(MathSpace::max_col_index(mat1, 0), 0); // First column maximum at row 0
    EXPECT_EQ(MathSpace::max_col_index(mat1, 1), 2); // First column maximum at row 2
    EXPECT_EQ(MathSpace::max_col_index(mat1, 2), 1); // First column maximum at row 2
} 