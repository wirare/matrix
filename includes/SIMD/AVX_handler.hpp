/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   AVX_handler.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/15 20:20:33 by wirare            #+#    #+#             */
/*   Updated: 2025/06/15 22:33:18 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <immintrin.h>

template<typename T>
struct AVX_struct;

template<>
struct AVX_struct<float> 
{
	using reg = __m256;

	static inline reg load(const float *src) { return _mm256_load_ps(src); }
	static inline void store(reg src, float *dest) { _mm256_store_ps(dest, src); }
	static inline reg set1(float x) { return _mm256_set1_ps(x); }
	static inline reg add(reg a, reg b) { return _mm256_add_ps(a, b); }
	static inline reg sub(reg a, reg b) { return _mm256_sub_ps(a, b); }
	static inline reg mul(reg a, reg b) { return _mm256_mul_ps(a, b); }

	static constexpr std::size_t width = 8;
};

template<>
struct AVX_struct<double>
{
	using reg = __m256d;

	static inline reg load(const double *src) { return _mm256_load_pd(src); }
	static inline void store(reg src, double *dest) { _mm256_store_pd(dest, src); }
	static inline reg set1(double x) { return _mm256_set1_pd(x); }
	static inline reg add(reg a, reg b) { return _mm256_add_pd(a, b); }
	static inline reg sub(reg a, reg b) { return _mm256_sub_pd(a, b); }
	static inline reg mul(reg a, reg b) { return _mm256_mul_pd(a, b); }

	static constexpr std::size_t width = 4;
};
