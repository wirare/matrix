/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Matrix.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/18 01:21:53 by wirare            #+#    #+#             */
/*   Updated: 2025/06/18 02:10:21 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include "../SIMD/Allocator.hpp"
#include "../SIMD/AVX_handler.hpp"
#include "../Math/Vector.hpp"
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <math.h>

template <typename K>
class Matrix
{
	public:
		Matrix(const std::vector<K> &vec, size_t rows, size_t cols): data(vec.begin(), vec.end()), rows(rows), cols(cols) {} 
		Matrix(size_t rows, size_t cols): data(rows * cols), rows(rows), cols(cols) {}
		Matrix(size_t rows, size_t cols, const K &value): data(rows * cols, value), rows(rows), cols(cols) {}
		Matrix(size_t rows, size_t cols, const std::initializer_list<K> &init): data(init), rows(rows), cols(cols) {}

		size_t size() const noexcept { return data.size(); }
		const aligned_vector<K> &getData() const noexcept { return data; }

		friend std::ostream& operator<<(std::ostream& os, const Matrix<K>& m) {
			os << "Matrix(" << m.rows << "x" << m.cols << ") [\n";
			for (size_t i = 0; i < m.rows; ++i) {
				os << "  [";
				if (m.cols > 0) {
					os << m.data[i * m.cols];
					for (size_t j = 1; j < m.cols; ++j) {
						os << ", " << m.data[i * m.cols + j];
					}
				}
				os << "]";
				if (i + 1 < m.rows) os << ",";
				os << "\n";
			}
			os << "]";
			return os;
		}

		Vector<K> operator()(size_t i) const 
		{
			if (i >= rows) throw std::out_of_range("Row index out of range");

			Vector<K> row(cols);
			const K* src = &data[i * cols];
			K* dest = row.data();

			const size_t w = AVX::width;
			const size_t chunks = cols / w;

			for (size_t j = 0; j < chunks; ++j) 
			{
				reg r = AVX::load(src + j * w);
				AVX::store(r, dest + j * w);
			}
			for (size_t j = w * chunks; j < cols; ++j) 
				dest[j] = src[j];

			return row;
		}
		
		K operator()(size_t i, size_t j) const 
		{
			if (i >= rows) throw std::out_of_range("Row index out of range");
			if (j >= cols) throw std::out_of_range("Column index out of range");

			return (data[j + i * cols]);
		}

	private:
		aligned_vector<K> data;
		size_t rows;
		size_t cols;
		using AVX = AVX_struct<K>;
		using reg = typename AVX::reg;
};

