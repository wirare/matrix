/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   AVX_handler_complex.hpp                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ellanglo <ellanglo@42angouleme.fr>         +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/21 10:35:11 by ellanglo          #+#    #+#             */
/*   Updated: 2025/06/21 14:28:46 by ellanglo         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma  once

#include "../Math/Complex.hpp"
#include <emmintrin.h>
#include <immintrin.h>
#include <math.h>

#define ALIGN alignas(32)

template<typename T>
struct AVX_struct;

template<>
struct AVX_struct<Complex<float>>
{
	using reg = __m256;
	using hreg = __m128;

	static inline reg set(const Complex<float> &z0, const Complex<float> &z1, const Complex<float> &z2, const Complex<float> &z3)
	{
		return _mm256_set_ps(z0.re, z0.im, z1.re, z1.im, z2.re, z2.im, z3.re, z3.im);
	}
	static inline reg load(const Complex<float> *src)
	{
		return set(src[0], src[1], src[2], src[3]);	
	}
	static inline void store(reg src, Complex<float>* dest)
	{
		ALIGN float temp[8];
		_mm256_store_ps(temp, src);

		dest[0] = Complex<float>(temp[0], temp[1]);
		dest[1] = Complex<float>(temp[2], temp[3]);
		dest[2] = Complex<float>(temp[4], temp[5]);
		dest[3] = Complex<float>(temp[6], temp[7]);
	}
	static inline void store(hreg src, Complex<float>* dest)
	{
		ALIGN float temp[4];
		_mm_store_ps(temp, src);

		dest[0] = Complex<float>(temp[0], temp[1]);
		dest[1] = Complex<float>(temp[2], temp[3]);
	}
	static inline reg set1(const Complex<float> &z)
	{
		return set(z, z, z, z);
	}
	static inline hreg extract_re(reg z)
	{
		hreg low = _mm256_castps256_ps128(z);
		hreg high = _mm256_extractf128_ps(z, 1);

		low = _mm_shuffle_ps(low, low, _MM_SHUFFLE(2, 0, 2, 0));
		high = _mm_shuffle_ps(high, high, _MM_SHUFFLE(2, 0, 2, 0));

		return _mm_movelh_ps(low, high);
	}
	static inline hreg extract_im(reg z)
	{
		hreg low = _mm256_castps256_ps128(z);
		hreg high = _mm256_extractf128_ps(z, 1);

		low = _mm_shuffle_ps(low, low, _MM_SHUFFLE(3, 1, 3, 1));
		high = _mm_shuffle_ps(high, high, _MM_SHUFFLE(3, 1, 3, 1));
	
		return _mm_movelh_ps(low, high);
	}
	static inline reg interleave_m128(hreg a, hreg b) 
	{
		hreg low  = _mm_unpacklo_ps(a, b);
		hreg high = _mm_unpackhi_ps(a, b);

		return _mm256_set_m128(high, low);
	}
	static inline reg add(reg z0, reg z1)
	{
		hreg z0_re = extract_re(z0);
		hreg z0_im = extract_im(z0);
		hreg z1_re = extract_re(z1);
		hreg z1_im = extract_im(z1);

		hreg res_re = _mm_add_ps(z0_re, z1_re);
		hreg res_im = _mm_add_ps(z0_im, z1_im);

		return interleave_m128(res_re, res_im);
	}
	static inline hreg add(hreg z0, hreg z1)
	{
		ALIGN Complex<float> z0_lst[2];
		ALIGN Complex<float> z1_lst[2];
		store(z0, z0_lst);
		store(z1, z1_lst);

		Complex<float> r0 = z0_lst[0] + z1_lst[0];
		Complex<float> r1 = z0_lst[1] + z1_lst[1];
		return _mm_set_ps(r0.re, r0.im, r1.re, r1.im);
	}
	static inline reg sub(reg z0, reg z1)
	{
		hreg z0_re = extract_re(z0);
		hreg z0_im = extract_im(z0);
		hreg z1_re = extract_re(z1);
		hreg z1_im = extract_im(z1);

		hreg res_re = _mm_sub_ps(z0_re, z1_re);
		hreg res_im = _mm_sub_ps(z0_im, z1_im);

		return interleave_m128(res_re, res_im);
	}	
	static inline reg mul(reg z0, reg z1)
	{
		hreg z0_re = extract_re(z0);
		hreg z0_im = extract_im(z0);
		hreg z1_re = extract_re(z1);
		hreg z1_im = extract_im(z1);

		hreg temp0 = _mm_mul_ps(z0_re, z1_re);
		hreg temp1 = _mm_mul_ps(z0_im, z1_im);
		hreg res_re = _mm_sub_ps(temp0, temp1);
		
		temp0 = _mm_mul_ps(z0_re, z1_im);
		temp1 = _mm_mul_ps(z0_im, z1_re);
		hreg res_im = _mm_add_ps(temp0, temp1);

		return interleave_m128(res_re, res_im);
	}
	static inline reg zero() { return _mm256_setzero_ps(); }
	static inline reg fmadd(reg z0, reg z1, reg z2) { return add(mul(z0, z1), z2); }
	static inline Complex<float> hsum(reg z)
	{
		hreg re = extract_re(z);
		hreg im = extract_im(z);

		hreg sum_re = _mm_hadd_ps(re, re);
		sum_re = _mm_hadd_ps(sum_re, sum_re);
		float res_re = _mm_cvtss_f32(sum_re);

		hreg sum_im = _mm_hadd_ps(im, im);
		sum_im = _mm_hadd_ps(sum_im, sum_im);
		float res_im = _mm_cvtss_f32(sum_im);

		return Complex<float>(res_re, res_im);
	}
	static inline Complex<float> hsum(hreg z)
	{
		hreg shuf = _mm_shuffle_ps(z, z, _MM_SHUFFLE(1, 0, 3, 2));
		hreg sum = _mm_add_ps(z, shuf);

		ALIGN Complex<float> res[2];

		store(sum, res);

		return res[0];
	}
	static inline reg max(reg z0, reg z1)
	{
		ALIGN Complex<float> z0_lst[4];
		ALIGN Complex<float> z1_lst[4];
		ALIGN Complex<float> res_lst[4];

		store(z0, z0_lst);
		store(z1, z1_lst);

		res_lst[0] = z0_lst[0] > z1_lst[0] ? z0_lst[0] : z1_lst[0];
		res_lst[1] = z0_lst[1] > z1_lst[1] ? z0_lst[1] : z1_lst[1];
		res_lst[2] = z0_lst[2] > z1_lst[2] ? z0_lst[2] : z1_lst[2];
		res_lst[3] = z0_lst[3] > z1_lst[3] ? z0_lst[3] : z1_lst[3];

		return load(res_lst);
	}
	static inline Complex<float> ext_max(reg z)
	{
		ALIGN Complex<float> z_lst[4];
		store(z, z_lst);

		Complex<float> res = z_lst[0];

		res = res > z_lst[1] ? res : z_lst[1];
		res = res > z_lst[2] ? res : z_lst[2];
		res = res > z_lst[3] ? res : z_lst[3];

		return res;
	}
	static inline reg i32gather(Complex<float> *base_addr, const int * indices)
	{
		__m128i indices_reg = _mm_load_si128(reinterpret_cast<const __m128i*>(indices));

		__m128i re_indices = _mm_slli_epi32(indices_reg, 1);
		__m128i im_indices = _mm_add_epi32(re_indices, _mm_set1_epi32(1));

		const float *float_addr = reinterpret_cast<const float*>(base_addr);

		hreg re = _mm_i32gather_ps(float_addr, re_indices, sizeof(float));
		hreg im = _mm_i32gather_ps(float_addr, im_indices, sizeof(float));

		return interleave_m128(re, im);
	}
	static inline hreg abs_(reg z)
	{
		ALIGN Complex<float> z_lst[4];
		ALIGN float res_float[4];
		store(z, z_lst);

		res_float[0] = z_lst[0].mag();
		res_float[1] = z_lst[1].mag();
		res_float[2] = z_lst[2].mag();
		res_float[3] = z_lst[3].mag();

		return _mm_load_ps(res_float);
	}

	static const constexpr std::size_t width = 4;
};

template<>
struct AVX_struct<Complex<double>>
{
	using reg = __m256d;
	using hreg = __m128d;

	static inline reg set(const Complex<double> &z0, const Complex<double> &z1)
	{
		return _mm256_set_pd(z0.re, z0.im, z1.re, z1.im);
	}
	static inline reg load(const Complex<double> *src)
	{
		return set(src[0], src[1]);
	}
	static inline void store(reg src, Complex<double>* dest)
	{
		ALIGN double temp[4];
		_mm256_store_pd(temp, src);

		dest[0] = Complex<double>(temp[0], temp[1]);
		dest[1] = Complex<double>(temp[2], temp[3]);
	}
	static inline void store(hreg src, Complex<double>* dest)
	{
		ALIGN double temp[2];
		_mm_store_pd(temp, src);

		dest[0] = Complex<double>(temp[0], temp[1]);
	}
	static inline reg set1(const Complex<double> &z)
	{
		return set(z, z);
	}
	static inline hreg extract_re(reg z)
	{
		hreg low = _mm256_castpd256_pd128(z);
		hreg high = _mm256_extractf128_pd(z, 1);

		return _mm_unpacklo_pd(low, high);
	}
	static inline hreg extract_im(reg z)
	{
		hreg low = _mm256_castpd256_pd128(z);
		hreg high = _mm256_extractf128_pd(z, 1);

		return _mm_unpacklo_pd(low, high);
	}
	static inline reg interleave_m128d(hreg a, hreg b)
	{
		hreg low = _mm_unpacklo_pd(a, b);
		hreg high = _mm_unpackhi_pd(a, b);

		return _mm256_set_m128d(high, low);
	}
	static inline reg add(reg z0, reg z1)
	{
		hreg z0_re = extract_re(z0);
		hreg z0_im = extract_im(z0);
		hreg z1_re = extract_re(z1);
		hreg z1_im = extract_im(z1);

		hreg res_re = _mm_add_pd(z0_re, z1_re);
		hreg res_im = _mm_add_pd(z0_im, z1_im);

		return interleave_m128d(res_re, res_im);
	}
	static inline reg sub(reg z0, reg z1)
	{
		hreg z0_re = extract_re(z0);
		hreg z0_im = extract_im(z0);
		hreg z1_re = extract_re(z1);
		hreg z1_im = extract_im(z1);

		hreg res_re = _mm_sub_pd(z0_re, z1_re);
		hreg res_im = _mm_sub_pd(z0_im, z1_im);

		return interleave_m128d(res_re, res_im);
	}	
	static inline reg mul(reg z0, reg z1)
	{
		hreg z0_re = extract_re(z0);
		hreg z0_im = extract_im(z0);
		hreg z1_re = extract_re(z1);
		hreg z1_im = extract_im(z1);

		hreg temp0 = _mm_mul_pd(z0_re, z1_re);
		hreg temp1 = _mm_mul_pd(z0_im, z1_im);
		hreg res_re = _mm_sub_pd(temp0, temp1);
		
		temp0 = _mm_mul_pd(z0_re, z1_im);
		temp1 = _mm_mul_pd(z0_im, z1_re);
		hreg res_im = _mm_add_pd(temp0, temp1);

		return interleave_m128d(res_re, res_im);
	}
	static inline reg zero() { return _mm256_setzero_pd(); }
	static inline reg fmadd(reg z0, reg z1, reg z2) { return add(mul(z0, z1), z2); }
	static inline Complex<double> hsum(reg z)
	{
		__m128d re = extract_re(z);
		__m128d im = extract_im(z);

		__m128d sum_re = _mm_hadd_pd(re, re);
		double res_re = _mm_cvtsd_f64(sum_re);

		__m128d sum_im = _mm_hadd_pd(im, im);
		double res_im = _mm_cvtsd_f64(sum_im);

		return Complex<double>(res_re, res_im);
	}
	static inline Complex<double> hsum(hreg z)
	{
		ALIGN Complex<double> res;
		store(z, &res);

		return res;
	}
	static inline reg max(reg z0, reg z1)
	{
		ALIGN Complex<double> z0_lst[2];
		ALIGN Complex<double> z1_lst[2];
		ALIGN Complex<double> res_lst[2];

		store(z0, z0_lst);
		store(z1, z1_lst);

		res_lst[0] = z0_lst[0] > z1_lst[0] ? z0_lst[0] : z1_lst[0];
		res_lst[1] = z0_lst[1] > z1_lst[1] ? z0_lst[1] : z1_lst[1];

		return load(res_lst);
	}
	static inline Complex<double> ext_max(reg z)
	{
		ALIGN Complex<double> z_lst[2];
		store(z, z_lst);

		Complex<double> res = z_lst[0] > z_lst[1] ? z_lst[0] : z_lst[1];

		return res;
	}
	static inline reg i32gather(Complex<double> *base_addr, const int *indices)
	{
		__m128i indices_reg = _mm_load_si128(reinterpret_cast<const __m128i*>(indices));

		__m128i re_indices = _mm_slli_epi32(indices_reg, 1);
		__m128i im_indices = _mm_add_epi32(re_indices, _mm_set1_epi32(1));

		const double *double_addr = reinterpret_cast<const double*>(base_addr);

		hreg re = _mm_i32gather_pd(double_addr, re_indices, sizeof(double));
		hreg im = _mm_i32gather_pd(double_addr, im_indices, sizeof(double));

		return interleave_m128d(re, im);
	}
	static inline hreg abs_(reg z)
	{
		ALIGN Complex<double> z_lst[2];
		ALIGN double res_double[2];
		store(z, z_lst);

		res_double[0] = z_lst[0].mag();
		res_double[1] = z_lst[1].mag();

		return _mm_load_pd(res_double);
	}

	static const constexpr std::size_t width = 2;
};
