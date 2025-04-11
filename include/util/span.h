#pragma once

/**
 * @file span.h
 * @brief Provides a fallback implementation of std::span for compilers that don't fully support C++20.
 * 
 * This implementation is used if the compiler doesn't provide <span> from the standard library.
 * It implements a minimal subset of std::span functionality needed for gQuery.
 */

#include <cstddef>
#include <array>
#include <type_traits>
#include <iterator>

namespace gquery {

// If C++20 span is not available, use our custom implementation
#if defined(__cpp_lib_span) && __cpp_lib_span >= 202002L
#include <span>
template<typename T>
using span = std::span<T>;
#else

template <class T>
class span {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using iterator = pointer;
    using const_iterator = const_pointer;

    // Constructors
    constexpr span() noexcept : data_(nullptr), size_(0) {}

    constexpr span(T* data, size_type size) noexcept : data_(data), size_(size) {}

    template <size_t N>
    constexpr span(T (&arr)[N]) noexcept : data_(arr), size_(N) {}

    template <size_t N>
    constexpr span(std::array<T, N>& arr) noexcept : data_(arr.data()), size_(N) {}

    template <class Container>
    constexpr span(Container& cont) : 
        data_(cont.data()), size_(cont.size()) {}

    // Element access
    constexpr reference operator[](size_type idx) const noexcept {
        return data_[idx];
    }

    constexpr pointer data() const noexcept { return data_; }

    // Iterators
    constexpr iterator begin() const noexcept { return data_; }
    constexpr iterator end() const noexcept { return data_ + size_; }
    constexpr const_iterator cbegin() const noexcept { return data_; }
    constexpr const_iterator cend() const noexcept { return data_ + size_; }

    // Observers
    constexpr size_type size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }

    // Subviews
    constexpr span subspan(size_type offset, size_type count) const {
        return span(data_ + offset, count);
    }

    constexpr span subspan(size_type offset) const {
        return span(data_ + offset, size_ - offset);
    }

    constexpr span first(size_type count) const {
        return span(data_, count);
    }

    constexpr span last(size_type count) const {
        return span(data_ + (size_ - count), count);
    }

private:
    pointer data_;
    size_type size_;
};

#endif // __cpp_lib_span check

} // namespace gquery 