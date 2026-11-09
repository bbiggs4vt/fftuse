
#include "stdafx.h"
#include "FftwWisdom.h"

#include <cstdio>
#include <cstdlib>
#include <fftw3.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>
#define FFT_GETPID _getpid
#else
#include <unistd.h>
#define FFT_GETPID getpid
#endif

namespace
{
/// Replaces `to` with `from`, atomically where the OS allows it
bool ReplaceFileAtomic(const std::string& from, const std::string& to)
{
#if defined(_WIN32)
	return MoveFileExA(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
	return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

/// Export without taking the lock (caller holds it)
std::string ExportLocked()
{
	char* text = fftwf_export_wisdom_to_string();
	if (text == nullptr)
	{
		return std::string();
	}
	std::string result(text);
	// FFTW allocates this with malloc(), not fftwf_malloc
	std::free(text);
	return result;
}
} // namespace

namespace FftwWisdom
{
bool Load(const std::string& path)
{
	dsp::utility::CFftwPlanLock lock;
	return fftwf_import_wisdom_from_filename(path.c_str()) != 0;
}

bool Save(const std::string& path)
{
	dsp::utility::CFftwPlanLock lock;
	const std::string temp = path + ".tmp" + std::to_string(FFT_GETPID());
	if (fftwf_export_wisdom_to_filename(temp.c_str()) == 0)
	{
		std::remove(temp.c_str());
		return false;
	}
	if (!ReplaceFileAtomic(temp, path))
	{
		std::remove(temp.c_str());
		return false;
	}
	return true;
}

std::string Export()
{
	dsp::utility::CFftwPlanLock lock;
	return ExportLocked();
}

bool Import(const std::string& wisdom)
{
	dsp::utility::CFftwPlanLock lock;
	return fftwf_import_wisdom_from_string(wisdom.c_str()) != 0;
}

void Forget()
{
	dsp::utility::CFftwPlanLock lock;
	fftwf_forget_wisdom();
}
} // namespace FftwWisdom

CFftwWisdomFile::CFftwWisdomFile(std::string path) :
	mPath(std::move(path)),
	mLoaded(false)
{
	mLoaded = FftwWisdom::Load(mPath);
	mSaved = FftwWisdom::Export();
}

CFftwWisdomFile::~CFftwWisdomFile()
{
	try
	{
		SaveIfChanged();
	}
	catch (...)
	{
		// Saving is best effort; never throw from a destructor
	}
}

bool CFftwWisdomFile::SaveIfChanged()
{
	std::string current = FftwWisdom::Export();
	if (current == mSaved)
	{
		return true;
	}
	if (!FftwWisdom::Save(mPath))
	{
		return false;
	}
	mSaved = std::move(current);
	return true;
}
