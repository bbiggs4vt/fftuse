#pragma once

// Stand-in precompiled header. The original Fft.h/Fft.cpp rely on this for
// standard headers, boost::shared_ptr and the FFTW plan lock.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "FftwPlanLock.h"
