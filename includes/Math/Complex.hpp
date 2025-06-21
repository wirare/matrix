/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Complex.hpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ellanglo <ellanglo@42angouleme.fr>         +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/18 17:51:25 by ellanglo          #+#    #+#             */
/*   Updated: 2025/06/21 14:38:21 by ellanglo         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <math.h>

template <typename K>
struct Complex
{
	K re;
	K im;

	Complex(): re(0), im(0) {}
	Complex(K re, K im): re(re), im(im) {}
	Complex(const Complex &z): re(z.re), im(z.im) {}
	Complex(K re): re(re), im(0) {}
	Complex& operator=(const Complex& other) = default;

	inline Complex &operator+(const Complex &z) 
	{
		re = re + z.re;
		im = im + z.im;
		return *this;
	}

	inline Complex &operator-(const Complex &z)
	{
		re = re - z.re;
		im = im - z.im;
		return *this;
	}

	inline Complex &operator*(const Complex &z)
	{
		re = re * z.re - im * z.im;
		im = re * z.im + im * z.re;
		return *this;
	}

	inline Complex &operator/(const Complex &z)
	{
		K denominator = z.re * z.re + z.im * z.im;
		re = (re * z.re + im * z.im) / denominator;
		im = (im * z.re - re * z.im) / denominator;
		return *this;
	}

	inline K mag() const { return std::pow((re * re + im * im), 0.5f); }

	inline bool operator>(const Complex &z) const { return (mag() > z.mag()); }
	inline bool operator<(const Complex &z) const { return (mag() < z.mag()); }
};
