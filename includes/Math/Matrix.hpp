/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Matrix.hpp                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/18 01:21:53 by wirare            #+#    #+#             */
/*   Updated: 2025/06/20 15:44:27 by ellanglo         ###   ########.fr       */
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

	Vector<K> extractCol(size_t i) const
	{
		if (i >= cols)
			throw std::out_of_range("Column index out of range");

		Vector<K> col(rows);
		K* dest = col.data();

		const size_t w = AVX::width;
		const size_t chunks = rows / w;

		for (size_t j = 0; j < chunks; ++j)
		{
			aligned_vector<K> indices(w);
			for (size_t k = 0; k < w; ++k)
				indices[k] = static_cast<int>((j * w + k) * cols + i);
			reg gather = AVX::i32gather(data.data(), indices.data());
			AVX::store(gather, dest + j * w);
		}

		for (size_t j = w * chunks; j < rows; ++j)
			dest[j] = data[j * cols + i];

		return col;
	}

	K operator()(size_t i, size_t j) const 
	{
		if (i >= rows) throw std::out_of_range("Row index out of range");
		if (j >= cols) throw std::out_of_range("Column index out of range");

		return (data[j + i * cols]);
	}

	void add(const Matrix &mat)
	{
		if (cols != mat.cols || rows != mat.rows) throw std::invalid_argument("Matrix addition requires sames shapes matrix");

		const size_t n = rows * cols;
		const size_t w = AVX::width;
		const size_t chunks =  n / w;

		K* a = data.data();
		const K* b = mat.data.data();

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

	void sub(const Matrix &mat)
	{
		if (cols != mat.cols || rows != mat.rows) throw std::invalid_argument("Matrix substraction requires sames shapes matrix");

		const size_t n = rows * cols;
		const size_t w = AVX::width;
		const size_t chunks =  n / w;

		K* a = data.data();
		const K* b = mat.data.data();

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
		const size_t n = rows * cols;
		const size_t w = AVX::width;
		const size_t chunks =  n / w;

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

	Vector<K> mul_vec(const Vector<K> &vec)
	{
		if (cols != vec.size()) throw std::invalid_argument("Vector dimension must be the same has Matrix rows size for multiplication");
		
		Vector<K> res(rows);

		for (size_t i = 0; i < rows; i++)
		{
			Vector<K> row = this(i);
			res[i] = row.dot(vec);
		}

		return res;
	}

	Matrix mul_mat(const Matrix &mat)
	{
		if (cols != mat.rows) throw std::invalid_argument("Matrix A must have the same number of cols than Matrix B has rows");
	
		Matrix<K> res(rows, mat.cols);

		for (size_t i = 0; i < rows; i++)
		{
			Vector<K> row = this(i);
			for (size_t j = 0; j < mat.cols; j++)
			{
				Vector<K> col = mat.extractCol(j);
				res(i, j) = row.dot(col);
			}
		}

		return res;
	}

	K trace()
	{
		if (cols != rows) throw std::invalid_argument("Matrix Trace computation require a square matrix");
	
		const size_t n = rows;
		const size_t w = AVX::width;
		const size_t chunks =  n / w;

		K *a = data.data();

		reg acc = AVX::zero();
		for (size_t i = 0; i < chunks; i++)
		{
			aligned_vector<K> indices(w);
			for (size_t k = 0; k < w; k++)
				indices[k] = static_cast<int>((i * w + k) * n + (i * w + k));
			reg gather = i32gather(a, indices.data());
			acc = AVX::add(gather, acc);
		}

		K res = AVX::hsum(acc);
		for (size_t i = chunks * w; i < n; i++)
			res += this(i, i);

		return res;
	}

private:
	aligned_vector<K> data;
	size_t rows;
	size_t cols;
	using AVX = AVX_struct<K>;
	using reg = typename AVX::reg;
};

