/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Vector.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/15 17:31:33 by wirare            #+#    #+#             */
/*   Updated: 2025/06/15 22:41:57 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include "../SIMD/Allocator.hpp"
#include "../SIMD/AVX_handler.hpp"
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

template <typename T>
class Vector
{
	public:
		Vector(const std::vector<T> &vec): data(vec.begin(), vec.end()) {}
		Vector(size_t size): data(size) {}
		Vector(size_t size, const T &value): data(size, value) {}

		size_t size() const noexcept { return data.size(); }
		const aligned_vector<T> &getData() const noexcept { return data; }

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

		T &operator[](size_t i) { return data[i]; }
		const T &operator[](size_t i) const { return data[i]; }

		void print(std::ostream& os = std::cout) const { os << *this << '\n'; }

		void resize(size_t new_size) { data.resize(new_size); }

		void add(const Vector &vec)
		{
			size_t n = size();

			if (n != vec.size()) throw std::invalid_argument("Vector addition requires equal sizes");
			
			size_t w = AVX::width;
			size_t chunks = n / w;

			T* a = data.data();
			const T* b = vec.getData().data();

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
			size_t n = size();

			if (n != vec.size()) throw std::invalid_argument("Vector substraction requires equal sizes");
			
			size_t w = AVX::width;
			size_t chunks = n / w;

			T* a = data.data();
			const T* b = vec.getData().data();

			for (size_t i = 0; i < chunks; i++)
			{
				reg r1 = AVX::load(a + i*w);
				reg r2 = AVX::load(b + i*w);
				reg r3 = AVX::sub(r1, r2);
				AVX::store(r3, a + i*w);
			}
			for (size_t i = w * chunks; i < n; i++)
				a[i] += b[i];
		}

		void scl(T x)
		{
			size_t n = size();
			size_t w = AVX::width;
			size_t chunks = n / w;

			T* a = data.data();

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

	private:
		aligned_vector<T>	data;
		using AVX = AVX_struct<T>;
		using reg = typename AVX::reg;
};
