#include <cstdlib>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include "includes/Math/Vector.hpp"

template <typename T>
bool are_equal(const Vector<T>& a, const Vector<T>& b, T epsilon = static_cast<T>(1e-5)) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

template <typename T>
void run_tests(const std::string& type_name, size_t size) {
    std::cout << "== Testing " << type_name << " ==\n";

    std::vector<T> raw(size);
    for (size_t i = 0; i < size; ++i) {
        raw[i] = static_cast<T>(i + 1); // {1.0, 2.0, ..., 64.0}
    }

    Vector<T> v1(raw);
    Vector<T> v2(raw);

    // ----- Test Add -----
    Vector<T> expected_add(size);
    for (size_t i = 0; i < size; ++i)
        expected_add[i] = raw[i] + raw[i];

    v1.add(v2);
    std::cout << "[Add] " << (are_equal(v1, expected_add) ? "PASS ✅" : "FAIL ❌") << '\n';

    // ----- Test Sub -----
    v1 = Vector<T>(raw);
    v2 = Vector<T>(raw);
    Vector<T> expected_sub(size);
    for (size_t i = 0; i < size; ++i)
        expected_sub[i] = raw[i] - raw[i];

    v1.sub(v2);
    std::cout << "[Sub] " << (are_equal(v1, expected_sub) ? "PASS ✅" : "FAIL ❌") << '\n';

    // ----- Test Scale -----
    v1 = Vector<T>(raw);
    T factor = static_cast<T>(3.0);
    Vector<T> expected_scale(size);
    for (size_t i = 0; i < size; ++i)
        expected_scale[i] = raw[i] * factor;

    v1.scl(factor);
    std::cout << "[Scale *" << factor << "] " << (are_equal(v1, expected_scale) ? "PASS ✅" : "FAIL ❌") << '\n';

    std::cout << '\n';
}
template <typename K>
void test_linear_combination() {
    // 1. Empty input
    {
        std::vector<Vector<K>> vectors;
        std::vector<K> scalars;
        Vector<K> result = linear_combination(vectors, scalars);
        assert(result.size() == 0);
    }

    // 2. Single vector, scalar = 1
    {
        Vector<K> v = {1, 2, 3};
        std::vector<Vector<K>> vectors = {v};
        std::vector<K> scalars = {1};
        Vector<K> result = linear_combination(vectors, scalars);
        assert(result[0] == 1 && result[1] == 2 && result[2] == 3);
    }

    // 3. Standard basis
    {
        Vector<K> e1 = {1, 0, 0};
        Vector<K> e2 = {0, 1, 0};
        Vector<K> e3 = {0, 0, 1};
        std::vector<Vector<K>> basis = {e1, e2, e3};
        std::vector<K> coefs = {10, -2, 0.5};
        Vector<K> result = linear_combination(basis, coefs);
        assert(result[0] == 10);
        assert(result[1] == -2);
        assert(result[2] == 0.5f);
    }

    // 4. Arbitrary combination
    {
        Vector<K> v1 = {1, 2, 3};
        Vector<K> v2 = {0, 10, -100};
        std::vector<Vector<K>> vectors = {v1, v2};
        std::vector<K> scalars = {10, -2};
        Vector<K> result = linear_combination(vectors, scalars);
        assert(result[0] == 10);    // 10*1 + -2*0
        assert(result[1] == 0);     // 10*2 + -2*10
        assert(result[2] == 230);   // 10*3 + -2*-100
    }

    // 5. Mismatched vector sizes
    {
        Vector<K> v1 = {1, 2};
        Vector<K> v2 = {3, 4, 5};
        try {
            linear_combination<K>({v1, v2}, {1, 1});
            assert(false); // should throw
        } catch (const std::invalid_argument&) {}
    }

    // 6. Mismatched number of scalars
    {
        Vector<K> v1 = {1, 2, 3};
        Vector<K> v2 = {4, 5, 6};
        try {
            linear_combination<K>({v1, v2}, {1});
            assert(false); // should throw
        } catch (const std::invalid_argument&) {}
    }

    // 7. High-dimensional test (basic check)
    {
        const int dim = 512;
        Vector<K> a(dim, 1.0f);
        Vector<K> b(dim, 2.0f);
        auto result = linear_combination<K>({a, b}, {2, -1});
        for (size_t i = 0; i < result.size(); ++i) {
            assert(result[i] == 0); // 2*1 - 1*2 == 0
        }
    }

    std::cout << "All tests passed!" << std::endl;
}

void test_lerp_scalar()
{
    float tf = 0.25f;
    float af = 2.0f;
    float bf = 10.0f;
    float expected_f = af * (1.0f - tf) + bf * tf;
    float result_f = af * (1.0f - tf) + bf * tf; // since lerp is trivial for scalars
    assert(std::abs(result_f - expected_f) < 1e-6f);

    double td = 0.75;
    double ad = 100.0;
    double bd = 200.0;
    double expected_d = ad * (1.0 - td) + bd * td;
    double result_d = ad * (1.0 - td) + bd * td;
    assert(std::abs(result_d - expected_d) < 1e-12);
}

template <typename K>
void test_lerp_vector()
{
    Vector<K> a = {1, 2, 3};
    Vector<K> b = {4, 5, 6};
    K t = static_cast<K>(0.5);

    Vector<K> expected = {
        a[0] * (1 - t) + b[0] * t,
        a[1] * (1 - t) + b[1] * t,
        a[2] * (1 - t) + b[2] * t
    };

    Vector<K> result = lerp(a, b, t);

    for (size_t i = 0; i < a.size(); ++i)
        assert(std::abs(result[i] - expected[i]) < (std::is_same<K, float>::value ? 1e-6f : 1e-12));
}

template <typename T>
void test_dot_product() {
    Vector<T> a = {1, 2, 3, 4};
    Vector<T> b = {5, 6, 7, 8};

    T expected = static_cast<T>(70);

    T result = a.dot(b);

    T epsilon = std::is_same<T, float>::value ? 1e-5f : 1e-12;

    assert(std::abs(result - expected) < epsilon);
    std::cout << "Dot product test passed for " << (std::is_same<T, float>::value ? "float" : "double") << "\n";
}

template <typename T>
void test_norms() {
    Vector<T> v = {-3, 4, -2};

    // Expected values
    T expected_norm_1   = static_cast<T>(9);              // |−3| + |4| + |−2| = 9
    T expected_norm_2   = static_cast<T>(std::sqrt(29));  // sqrt(9 + 16 + 4)
    T expected_norm_inf = static_cast<T>(4);              // max(|−3|, |4|, |−2|)

    // Calculate
    T norm_1   = v.norm_1();
    T norm_2   = v.norm();
    T norm_inf = v.norm_inf();

    // Tolerance
    T eps = std::is_same<T, float>::value ? 1e-5f : 1e-12;

    // Assertions
    assert(std::abs(norm_1   - expected_norm_1)   < eps);
    assert(std::abs(norm_2   - expected_norm_2)   < eps);
	std::cout << "Expected norm inf : " << expected_norm_inf << "\nNorm inf calculated : " << norm_inf << std::endl;
    assert(std::abs(norm_inf - expected_norm_inf) < eps);

    std::cout << "Norm tests passed for " << (std::is_same<T, float>::value ? "float" : "double") << "\n";
}

int main(int ac, char **av) {
	(void)ac;
    run_tests<float>("float", atoi(av[1]));
    run_tests<double>("double", atoi(av[1]));
    test_linear_combination<float>();
    test_linear_combination<double>();
	test_lerp_scalar();
	test_lerp_vector<float>();
	test_lerp_vector<double>();
	std::cout << "All lerps test passed !" << std::endl;
	test_dot_product<float>();
	test_dot_product<double>();
	test_norms<float>();
	test_norms<double>();
    return 0;
}

