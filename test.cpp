#include <iostream>
#include <vector>
#include <cmath>
#include "includes/Vectors/Vector.hpp" // Make sure this path is correct

template <typename T>
bool are_equal(const Vector<T>& a, const Vector<T>& b, T epsilon = static_cast<T>(1e-5)) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::abs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

template <typename T>
void run_tests(const std::string& type_name) {
    std::cout << "== Testing " << type_name << " ==\n";
    size_t size = 256; // larger size for AVX batch processing

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

int main() {
    run_tests<float>("float");
    run_tests<double>("double");
    return 0;
}

