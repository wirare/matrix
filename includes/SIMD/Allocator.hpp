/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   Allocator.hpp                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: wirare <wirare@42angouleme.fr>             +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/06/15 17:51:30 by wirare            #+#    #+#             */
/*   Updated: 2025/06/15 18:40:20 by wirare           ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
#pragma once

#include <cstddef>
#include <new>
#include <cstdlib>
#include <vector>

template<typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept {}
    template<class U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)))
            throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }

	template<typename U>
	struct rebind {
		using other = AlignedAllocator<U, Alignment>;
	};
};

template<typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator==(AlignedAllocator<T1, A1> const&,
                AlignedAllocator<T2, A2> const&) noexcept {
    return A1 == A2;
}
template<typename T1, std::size_t A1, typename T2, std::size_t A2>
bool operator!=(AlignedAllocator<T1, A1> const& a,
                AlignedAllocator<T2, A2> const& b) noexcept {
    return !(a == b);
}

template<typename K>
using aligned_vector = std::vector<K, AlignedAllocator<K, 32>>;
