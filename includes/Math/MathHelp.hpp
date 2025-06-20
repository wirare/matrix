/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   MathHelp.hpp                                       :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/20 11:21:57 by wirare            #+#    #+#             */
/*   Updated: 2025/06/20 13:43:48 by ellanglo         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include "../Math/Complex.hpp"
#include <algorithm>
#include <math.h>

template<typename T>
struct MathHelp;

template<>
struct MathHelp<float>
{
	static inline float max(float a, float b) { return std::max(a, b); }
	static inline float sqrt(float a) { return std::pow(a, 0.5f); }
	static inline float abs_(float a) { return abs(a); }
};

template<>
struct MathHelp<double>
{
	static inline double max(double a, double b) { return std::max(a, b); }
	static inline double sqrt(double a) { return std::pow(a, 0.5f); }
	static inline double abs_(double a) { return abs(a); }
};

template<typename T>
struct MathHelp<Complex<T>>
{
	static inline Complex<T> &max(const Complex<T> &a, const Complex<T> &b)
	{
		if (a.mag() > b.mag())
			return a;
		return b;
	}	
	static inline Complex<T> sqrt(const Complex<T> &a)
	{ 
		Complex<T> res;
		T r = a.mag();
		int sign = a.im < 0 ? -1 : 1;
		res.re = std::pow((r + a.re)/2, 0.5f);
		res.im = std::pow((r - a.im)/2, 0.5f) * sign;
		return res;
	}
	static inline T abs_(const Complex<T> &a) { return a.mag(); }
};
