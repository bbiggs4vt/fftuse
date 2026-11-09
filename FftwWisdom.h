
#pragma once

// Saving and loading FFTW wisdom: the planner's record of which algorithms were
// fastest on this machine. With wisdom loaded, a CFftV2/V3/V4 built with
// PlanEffort::Measure plans in well under a millisecond, not 0.1-2 s, and one
// built with PlanEffort::Estimate also gets the measured plan.
//
// Things to know:
//  - Wisdom is specific to the machine (CPU) and to the FFTW version. Keep the
//    file per machine, for example in a per-user cache directory, not in source
//    control. FFTW rejects wisdom from a different version (Load returns false),
//    but it can't detect a different CPU: stale wisdom still works, it is just
//    not tuned.
//  - Wisdom is specific to the plan's shape: size, direction, in-place or
//    out-of-place, and flags. v2 (in-place) and v4 (out-of-place) need separate
//    entries, so generate wisdom with the class the application uses (see
//    tools/fft_wisdom.cpp).
//  - Wisdom from Measure serves Measure and Estimate plans. Patient plans need
//    wisdom recorded with Patient. FFTW also records Estimate plans as wisdom,
//    but those entries never stand in for a Measure plan, so they are harmless.
//  - Wisdom is global FFTW state. Every function here holds CFftwPlanLock, so it
//    is safe to call while other threads create or destroy CFft objects.

#include <string>

namespace FftwWisdom
{
/// Merges wisdom from a file into FFTW's wisdom. Returns false if the file is
/// missing, unreadable, or not valid wisdom for this FFTW version. FFTW's
/// wisdom is unchanged on failure.
bool Load(const std::string& path);

/// Writes all of FFTW's current wisdom to a file. Writes a temporary file next
/// to it and renames it into place, so a crash or a concurrent reader never sees
/// a partial file. Returns false on failure.
bool Save(const std::string& path);

/// FFTW's current wisdom as text (empty if FFTW can't export it)
std::string Export();

/// Merges wisdom from text. Returns false if it isn't valid wisdom.
bool Import(const std::string& wisdom);

/// Discards all wisdom held in memory. Plans that already exist are unaffected.
void Forget();
} // namespace FftwWisdom

/// Loads a wisdom file on construction and saves it on destruction if new
/// wisdom was gathered in between, for example because a Measure plan was
/// created for a new size. Typical use is one instance in main(), constructed
/// before any CFft objects:
///
///     CFftwWisdomFile wisdom(cacheDir + "/fftwf.wisdom");
///     ... create CFftV4 objects with PlanEffort::Measure ...
///
/// The first run pays the measurement cost and writes the file. Later runs load
/// it and plan instantly.
class CFftwWisdomFile
{
public:
	explicit CFftwWisdomFile(std::string path);

	/// Saves if wisdom changed. Never throws.
	~CFftwWisdomFile();

	CFftwWisdomFile(const CFftwWisdomFile&) = delete;
	CFftwWisdomFile& operator=(const CFftwWisdomFile&) = delete;

	/// Whether the file existed and loaded
	bool Loaded() const { return mLoaded; }

	/// Saves now if wisdom changed since the last load or save. Returns false
	/// only if a save was needed and failed.
	bool SaveIfChanged();

	const std::string& Path() const { return mPath; }

private:
	std::string mPath;
	std::string mSaved; //!< wisdom text as of the last load or save
	bool mLoaded;
};
