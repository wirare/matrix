/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Vector.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/15 17:31:33 by wirare            #+#    #+#             */
/*   Updated: 2025/06/21 15:39:36 by ellanglo         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include "../SIMD/Allocator.hpp"
#include "../SIMD/AVX_handler.hpp"
#include "MathHelp.hpp"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <math.h>

template <typename K>
class Vector;

template<typename K>
static Vector<K> linear_combination(const std::vector<Vector<K>> &vectors, const std::vector<K> &scalars)
{
	using AVX = AVX_struct<K>;
	using reg = typename AVX::reg;

	if (vectors.size() != scalars.size()) throw std::invalid_argument("Linear combination requires equal sizes on vectors and scalars");

	if (vectors.empty())
		return Vector<K>(0);

	const size_t size = vectors.size();
	const size_t dimension = vectors[0].size();

	for (const auto &v : vectors)
		if (v.size() != dimension)
			throw std::invalid_argument("Linear combination requires every vector to be the same dimension");

	Vector<K> res(dimension);
	const size_t w = AVX::width;

	size_t i = 0;
	for (; i + w - 1 < dimension; i += w)
	{
		reg temp = AVX::zero();
		for (size_t j = 0; j < size; j++)
		{
			reg r1 = AVX::load(&vectors[j].data[i]); 
			reg r2 = AVX::set1(scalars[j]);
			temp = AVX::fmadd(r1, r2, temp);
		}
		AVX::store(temp, &res.data[i]);
	}

	for (; i < dimension; i++)
	{
		K temp = 0;
		for (size_t j = 0; j < size; j++)
			temp += vectors[j].data[i] * scalars[j];
		res.data[i] = temp;
	}

	return res;
}

template <typename K>
static Vector<K> lerp(const Vector<K> &vec_a, const Vector<K> &vec_b, K t)
{
	using AVX = AVX_struct<K>;
	using reg = typename AVX::reg;

	const size_t size = vec_a.size();

	if (size != vec_b.size()) throw std::invalid_argument("Vector lerp require equal vector size");

	Vector<K> res(size);

	const size_t w = AVX::width;
	const size_t chunks = size / w;

	const K* a = vec_a.data.data();
	const K* b = vec_b.data.data();

	reg reg_t = AVX::set1(t);
	reg reg_one_minus_t = AVX::sub(AVX::set1(static_cast<K>(1)), reg_t);
	for (size_t i = 0; i < chunks; i++)
	{
		reg r1 = AVX::load(a + i*w);
		reg r2 = AVX::load(b + i*w);
		reg r3 = AVX::mul(r2, reg_t);
		reg r4 = AVX::fmadd(r1, reg_one_minus_t, r3);
		AVX::store(r4, res.data.data() + i*w);
	}
	for (size_t i = w * chunks; i < size; i++)
		res[i] = a[i] * (1 - t) + (b[i] * t);
	return res;
}

template <typename K>
static K angle_cos(const Vector<K> &a, const Vector<K> &b)
{
	if (a.size() != b.size()) throw std::invalid_argument("Vector cosine require equal vector size");

	K dot = a.dot(b);
	K norm_product = a.norm() * b.norm();
	K res = dot / norm_product;
	
	return res;
}

template <typename K>
static Vector<K> cross_product(const Vector<K> &a, const Vector<K> &b)
{
	if (a.size() != 3 || b.size() != 3)
		throw std::invalid_argument("Cross product require 2 3-dimentional vector");

	#define X 0
	#define Y 1
	#define Z 2
	Vector<K> res = 
	{
		a[Y] * b[Z] - a[Z] * b[Y],
		a[Z] * b[X] - a[X] * b[Z],
		a[X] * b[Y] - a[Y] * b[X]
	};
	#undef X
	#undef Y
	#undef Z

	return res;
}

template <typename K>
class Matrix;

template <typename K>
class Vector
{
public:
	Vector(const std::vector<K> &vec): data(vec.begin(), vec.end()) {}
	Vector(size_t size): data(size) {}
	Vector(size_t size, const K &value): data(size, value) {}
	Vector(const std::initializer_list<K> &init): data(init) {}

	size_t size() const noexcept { return data.size(); }
	const aligned_vector<K> &getData() const noexcept { return data; }

	friend std::ostream& operator<<(std::ostream& os, Vector const& v) {
		os << "Vector(size=" << v.size() << ") [";
		if (!v.data.empty()) {
			os << v.data.front();
			for (std::size_t i = 1; i < v.data.size(); ++i) {
				os << ", " << v.data[i];
			}
		}
		os << "]";
		return os;
	}

	K &operator[](size_t i) { return data[i]; }
	const K &operator[](size_t i) const { return data[i]; }

	void print(std::ostream& os = std::cout) const { os << *this << '\n'; }

	void resize(size_t new_size) { data.resize(new_size); }

	void add(const Vector &vec)
	{
		const size_t n = size();

		if (n != vec.size()) throw std::invalid_argument("Vector addition requires equal sizes");

		const size_t w = AVX::width;
		const size_t chunks = n / w;

		K* a = data.data();
		const K* b = vec.data.data();

		for (size_t i = 0; i < chunks; i++)
		{
			reg r1 = AVX::load(a + i*w);
			reg r2 = AVX::load(b + i*w);
			reg r3 = AVX::add(r1, r2);
			AVX::store(r3, a + i*w);
		}
		for (size_t i = w * chunks; i < n; i++)
			a[i] += b[i];
	}	

	void sub(const Vector &vec)
	{
		const size_t n = size();

		if (n != vec.size()) throw std::invalid_argument("Vector substraction requires equal sizes");

		const size_t w = AVX::width;
		const size_t chunks = n / w;

		K* a = data.data();
		const K* b = vec.data.data();

		for (size_t i = 0; i < chunks; i++)
		{
			reg r1 = AVX::load(a + i*w);
			reg r2 = AVX::load(b + i*w);
			reg r3 = AVX::sub(r1, r2);
			AVX::store(r3, a + i*w);
		}
		for (size_t i = w * chunks; i < n; i++)
			a[i] -= b[i];
	}

	void scl(K x)
	{
		const size_t n = size();
		const size_t w = AVX::width;
		const size_t chunks = n / w;

		K* a = data.data();

		reg scalar = AVX::set1(x);
		for (size_t i = 0; i < chunks; i++)
		{
			reg r1 = AVX::load(a + i*w);
			reg r2 = AVX::mul(r1, scalar);
			AVX::store(r2, a + i*w);
		}
		for (size_t i = w * chunks; i < n; i++)
			a[i] *= x;
	}

	K dot(const Vector &vec)
	{
		const size_t n = size();

		if (n != vec.size()) throw std::invalid_argument("Dot product requires equal sizes");

		const size_t w = AVX::width;
		const size_t chunks = n / w;

		K* a = data.data();
		const K* b = vec.data.data();

		reg acc = AVX::zero();
		for (size_t i = 0; i < chunks; i++)
		{
			reg r1 = AVX::load(a + i*w);
			reg r2 = AVX::load(b + i*w);
			acc = AVX::fmadd(r1, r2, acc);
		}

		K res = AVX::hsum(acc);

		for (size_t i = w * chunks; i < n; i++)
			res += a[i] * b[i];

		return res;
	}

	K norm_1()
	{
		const size_t n = size();
		const size_t w = AVX::width;
		const size_t chunks = n / w;

		K* a = data.data();

		K res;

		if constexpr (!is_complex<K>::value)
		{
			reg acc = AVX::zero();
			for (size_t i = 0; i < chunks; i++)
			{
				reg r1 = AVX::load(a + i*w);
				reg r2 = AVX::abs_(r1);
				acc = AVX::add(r2, acc);
			}
			res = AVX::hsum(acc);
		}
		else
		{
			using hreg = typename AVX::hreg;
			
			hreg acc = AVX::zero();
			for (size_t i = 0; i < chunks; i++)
			{
				reg r1 = AVX::load(a + i*w);
				hreg r2 = AVX::abs_(r1);
				acc = AVX::add(r2, acc);
			}
			res = AVX::hsum(acc);
		}
		
		for (size_t i = w * chunks; i < n; i++)
			res += Math::abs_(a[i]);

		return res;
	}

	K norm()
	{
		const size_t n = size();
		const size_t w = AVX::width;
		const size_t chunks = n / w;

		K* a = data.data();

		reg acc = AVX::zero();
		for (size_t i = 0; i < chunks; i++)
		{
			reg r1 = AVX::load(a + i*w);
			reg r2 = AVX::mul(r1, r1);
			acc = AVX::add(r2, acc);
		}

		K res = AVX::hsum(acc);

		K tmp;
		for (size_t i = w * chunks; i < n; i++)
		{
			tmp = a[i];
			res += tmp * tmp;
		}

		return Math::sqrt(res);
	}

	K norm_inf()
	{
		const size_t n = size();
		const size_t w = AVX::width;
		const size_t chunks = n / w;

		K* a = data.data();

		reg r_max = AVX::zero();
		for (size_t i = 0; i < chunks; i++)
		{
			reg r1 = AVX::load(a + i*w);
			reg r2 = AVX::abs_(r1);
			r_max = AVX::max(r2, r_max);
		}

		K res = AVX::ext_max(r_max);

		for (size_t i = w * chunks; i < n; i++)
			res = Math::max(res, Math::abs_(a[i]));
		
		return res;
	}

	Vector &operator+(const Vector& vec) {return add(vec); }
	Vector &operator-(const Vector& vec) {return sub(vec); }
	Vector &operator*(K scalar) { return scl(scalar); }

	friend Vector<K> linear_combination<>(const std::vector<Vector<K>> &vectors, const std::vector<K> &scalars);
	friend Vector<K> lerp<>(const Vector<K> &a, const Vector<K> &b, K t);
	friend K angle_cos<>(const Vector<K> &a, const Vector<K> &b);
	friend Vector<K> cross_product<>(const Vector<K> &a, const Vector<K> &b);

private:
	aligned_vector<K> data;
	using AVX = AVX_struct<K>;
	using reg = typename AVX::reg;
	using Math = MathHelp<K>;
};
