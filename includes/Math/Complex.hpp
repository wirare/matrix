/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Complex.hpp                                        :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: ellanglo <ellanglo@42angouleme.fr>         +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/18 17:51:25 by ellanglo          #+#    #+#             */
/*   Updated: 2025/06/18 18:34:56 by ellanglo         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

template <typename K>
struct Complex
{
	K re;
	K im;

	Complex(K re, K im): re(re), im(im) {}
	Complex(const Complex &z): re(z.re), im(z.im) {}
	Complex(K re): re(re), im(0) {}

	inline Complex &operator+(const Complex &z) 
	{
		Complex res;
		res.re = re + z.re;
		res.im = im + z.im;
		return res;
	}
	inline Complex &operator-(const Complex &z)
	{
		Complex res;
		res.re = re - z.re;
		res.im = im - z.im;
		return res;
	}
	inline Complex &operator*(const Complex &z)
	{
		Complex res;
		res.re = re * z.re - im * z.im;
		res.im = re * z.im + im * z.re;
		return res;
	}
	inline Complex &operator/(const Complex &z)
	{
		Complex res;
		K denominator = z.re * z.re + z.im * z.im;
		res.re = (re * z.re + im * z.im) / denominator;
		res.im = (im * z.re - re * z.im) / denominator;
		return res;
	}
};
