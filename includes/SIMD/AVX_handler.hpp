/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   AVX_handler.hpp                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/15 20:20:33 by wirare            #+#    #+#             */
/*   Updated: 2025/06/17 16:12:51 by wirare           ###   ########.fr       */
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
	static inline float hsum(__m256 a) 
	{
		__m128 low  = _mm256_castps256_ps128(a);
		__m128 high = _mm256_extractf128_ps(a, 1);
		__m128 sum = _mm_add_ps(low, high);
		sum = _mm_hadd_ps(sum, sum);
		sum = _mm_hadd_ps(sum, sum);
		return _mm_cvtss_f32(sum);
	}
	static inline reg and_(reg a, reg b) { return _mm256_and_ps(a, b); }
	static inline reg sqrt(reg a) { return _mm256_sqrt_ps(a); }
	static inline reg max(reg a, reg b) { return _mm256_max_ps(a, b); }
	static inline float ext_max(reg a) {
		__m128 low  = _mm256_castps256_ps128(a);
		__m128 high = _mm256_extractf128_ps(a, 1);
		__m128 max128 = _mm_max_ps(low, high);
		__m128 temp = _mm_movehdup_ps(max128);
		max128 = _mm_max_ps(max128, temp);
		temp = _mm_movehl_ps(temp, max128);
		max128 = _mm_max_ss(max128, temp);
		return _mm_cvtss_f32(max128);
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
	static inline double hsum(__m256d a) 
	{
		__m128d low  = _mm256_castpd256_pd128(a);
		__m128d high = _mm256_extractf128_pd(a, 1);
		__m128d sum = _mm_add_pd(low, high);
		double r[2];
		_mm_store_pd(r, sum);
		return r[0] + r[1];
	}
	static inline reg and_(reg a, reg b) { return _mm256_and_pd(a, b); }
	static inline reg sqrt(reg a) { return _mm256_sqrt_pd(a); }
	static inline reg max(reg a, reg b) { return _mm256_max_pd(a, b); }
	static inline double ext_max(reg a) 
	{
		__m128d low  = _mm256_castpd256_pd128(a);
		__m128d high = _mm256_extractf128_pd(a, 1);
		__m128d max2 = _mm_max_pd(low, high);
		__m128d shuf = _mm_unpackhi_pd(max2, max2);
		__m128d max1 = _mm_max_sd(max2, shuf);
		return _mm_cvtsd_f64(max1);
	}

	static const constexpr std::size_t width = 4;
};
