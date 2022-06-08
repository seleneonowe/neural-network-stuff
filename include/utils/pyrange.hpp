/*
This file (and only this file) is licensed under the MIT License

Copyright (c) 2019-2022 Richard Spencer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdexcept>
 
template<typename T = int>
class pyrange {
    struct end_range {
        T e;
    };
    struct begin_range {
        T b, s;
        begin_range& operator++() { b += s; return *this; }
        bool operator!=(const end_range& c) const
            { return (s < 0) ? b > c.e : b < c.e; }
        const T& operator*() const { return b; }
    };
    begin_range br;
    end_range er;
public:
    explicit pyrange(T arg_e) : br{ 0, 1 }, er{ arg_e } {}
    explicit pyrange(T arg_b, T arg_e, T arg_s = 1) : br{ arg_b, arg_s }, er{ arg_e }
    { if (!arg_s) throw std::out_of_range{"pyrange: step must be non-zero"}; }
    auto& begin() { return br; }
    auto& end() { return er; }
};