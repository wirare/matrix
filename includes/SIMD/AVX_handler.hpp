/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   AVX_handler.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/15 20:20:33 by wirare            #+#    #+#             */
/*   Updated: 2025/06/16 15:27:20 by ellanglo         ###   ########.fr       */
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
	static inline reg zero() { return _mm256_setzero_ps(); }
	static inline reg fmadd(reg a, reg b, reg c) { return _mm256_fmadd_ps(a, b ,c); }
	static inline float hsum(reg a) {
		__m128 low  = _mm256_castps256_ps128(a);
		__m128 high = _mm256_extractf128_ps(a, 1);
		__m128 sum128 = _mm_add_ps(low, high);
		__m128 shuf = _mm_movehdup_ps(sum128);
		__m128 sums = _mm_add_ps(sum128, shuf);
		shuf = _mm_movehl_ps(shuf, sums);
		sums = _mm_add_ss(sums, shuf);
		return _mm_cvtss_f32(sums);
	}

	static const constexpr std::size_t width = 8;
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
	static inline reg zero() { return _mm256_setzero_pd(); }
	static inline reg fmadd(reg a, reg b, reg c) { return _mm256_fmadd_pd(a, b ,c); }
	static inline double hsum(reg a) {
		__m128d low  = _mm256_castpd256_pd128(a);
		__m128d high = _mm256_extractd128_pd(a, 1);
		__m128d sum128 = _mm_add_pd(low, high);
		__m128d shuf = _mm_movehdup_pd(sum128);
		__m128d sums = _mm_add_pd(sum128, shuf);
		shuf = _mm_movehl_pd(shuf, sums);
		sums = _mm_add_ss(sums, shuf);
		return _mm_cvtss_f32(sums);
	}

	static const constexpr std::size_t width = 4;
};
