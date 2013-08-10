#include <array>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "ba-file.h"
#include "compiler.h"
#include "opt.h"
#include "SysTools.h"

typedef std::array<uint64_t,3> UINT64VECTOR3;
typedef std::array<uint64_t,4> UINT64VECTOR4;
typedef std::array<unsigned,3> UINTVECTOR3;

class BrickAccessFile {
public:
  BrickAccessFile(std::string const& filename);
  BrickAccessFile(std::wstring const& filename);

  typedef UINT64VECTOR4 Brick;
  typedef std::vector<Brick> Subframe;
  typedef std::vector<Subframe> Frame;

  bool Load();
  std::string const& GetUVFFilename() const { return m_Filename; }
  UINTVECTOR3 const& GetMaxBrickSize() const { return m_MaxBrickSize; }
  UINTVECTOR3 const& GetBrickOverlap() const { return m_BrickOverlap; }
  uint64_t GetLoDCount() const { return (uint64_t)m_DomainSizes.size(); }
  std::vector<UINT64VECTOR3> const& GetDomainSizes() const { return m_DomainSizes; }
  std::vector<UINT64VECTOR3> const& GetBrickCounts() const { return m_BrickCounts; }
  std::vector<Frame> const& GetFrames() const { return m_Frames; }

private:
  std::ifstream m_File;
  std::string m_Filename;
  UINTVECTOR3 m_MaxBrickSize;
  UINTVECTOR3 m_BrickOverlap;
  std::vector<UINT64VECTOR3> m_DomainSizes;
  std::vector<UINT64VECTOR3> m_BrickCounts;
  std::vector<Frame> m_Frames;
};

BrickAccessFile::BrickAccessFile(std::string const& filename)
  : m_File(filename)
{}

#if 0
BrickAccessFile::BrickAccessFile(std::wstring const& filename)
  : m_File(filename)
{}
#endif

bool BrickAccessFile::Load()
{
  if (!m_File.is_open()) return false;

  m_Frames.clear();
  uint32_t iExpectedBricks = 0;
  uint32_t iSubframeCounter = 0;
  uint32_t iFrameCounter = 0;

  std::vector<std::string> tokens, elements;
  std::string line;
  for (uint32_t i=1; getline(m_File, line); ++i) {
    if (line.size() > 0) {
      if (line[0] == '#' )
        continue; // skip comment line

      if (line[0] == '[') {
        Frame& frame = m_Frames.back();
        Subframe& subframe = frame.back();

        tokens = SysTools::Tokenize(line, SysTools::PM_BRACKETS, '[', ']');
        if (tokens.size() != iExpectedBricks) {
          std::cerr << "failed to parse line " << i << ": wrong brick count for subframe" << std::endl;
          return false;
        }
        for (size_t t=0; t<tokens.size(); ++t) {
          elements = SysTools::Tokenize(tokens[t], SysTools::PM_NONE);
          if (elements.size() != 4) {
            std::cerr << "failed to parse line " << i << " at brick number: " << t << std::endl;
            return false;
          }
          subframe.push_back(Brick());
          Brick& brick = subframe.back();
          brick[0] = SysTools::FromString<uint32_t>(elements[0]);
          brick[1] = SysTools::FromString<uint32_t>(elements[1]);
          brick[2] = SysTools::FromString<uint32_t>(elements[2]);
          brick[3] = SysTools::FromString<uint32_t>(elements[3]);
          if (brick[3] >= m_BrickCounts.size()) {
            std::cerr << "failed to parse line " << i << " at brick number: " << t << " brick LoD exceeds bounds" << std::endl;
            return false;
          }
          if (brick[0] >= m_BrickCounts[brick[3]][0] ||
              brick[1] >= m_BrickCounts[brick[3]][1] ||
              brick[2] >= m_BrickCounts[brick[3]][2]) {
            std::cerr << "failed to parse line " << i << " at brick number: " << t << " brick position exceeds bounds" << std::endl;
            return false;
          }
        }
        iSubframeCounter++;
        continue; // finished parsing subframe
      }

      // parse header and subframe and frame marks
      tokens = SysTools::Tokenize(line, SysTools::PM_CUSTOM_DELIMITER, '=');
      if (tokens.size() > 1) {
        if (tokens[0] == "Filename") {
          m_Filename = tokens[1];

        } else if (tokens[0] == "MaxBrickSize") {
          tokens = SysTools::Tokenize(tokens[1], SysTools::PM_NONE);
          if (tokens.size() != 3) {
            std::cerr << "failed to parse line " << i << ": invalid MaxBrickSize" << std::endl;
            return false;
          }
          m_MaxBrickSize[0]= SysTools::FromString<uint32_t>(tokens[0]);
          m_MaxBrickSize[1]= SysTools::FromString<uint32_t>(tokens[1]);
          m_MaxBrickSize[2]= SysTools::FromString<uint32_t>(tokens[2]);

        } else if (tokens[0] == "BrickOverlap") {
          tokens = SysTools::Tokenize(tokens[1], SysTools::PM_NONE);
          if (tokens.size() != 3) {
            std::cerr << "failed to parse line " << i << ": invalid BrickOverlap" << std::endl;
            return false;
          }
          m_BrickOverlap[0]= SysTools::FromString<uint32_t>(tokens[0]);
          m_BrickOverlap[1]= SysTools::FromString<uint32_t>(tokens[1]);
          m_BrickOverlap[2]= SysTools::FromString<uint32_t>(tokens[2]);

        } else if (tokens[0] == "LoDCount") {
          if (tokens.size() != 2) {
            std::cerr << "failed to parse line " << i << ": invalid LoDCount" << std::endl;
            return false;
          }
          uint32_t const LoDCount = SysTools::FromString<uint32_t>(tokens[1]);
          m_DomainSizes.resize(LoDCount);
          m_BrickCounts.resize(LoDCount);

        } else if (tokens[0] == " LoD" || tokens[0] == "LoD") {
          if (tokens.size() != 4) {
            std::cerr << "failed to parse line " << i << ": invalid LoD" << std::endl;
            return false;
          }
          elements = SysTools::Tokenize(tokens[1], SysTools::PM_NONE);
          if (elements.size() != 2) {
            std::cerr << "failed to parse line " << i << ": invalid LoD value" << std::endl;
            return false;
          }
          uint32_t const LoD = SysTools::FromString<uint32_t>(elements[0]);
          if (LoD >= m_DomainSizes.size() || LoD >= m_BrickCounts.size()) {
            std::cerr << "failed to parse line " << i << ": LoD not available" << std::endl;
            return false;
          }
          if (elements[1] != "DomainSize") {
            std::cerr << "failed to parse line " << i << ": DomainSize not available" << std::endl;
            return false;
          }
          elements = SysTools::Tokenize(tokens[2], SysTools::PM_NONE);
          if (elements.size() != 4) {
            std::cerr << "failed to parse line " << i << ": invalid DomainSize" << std::endl;
            return false;
          }
          m_DomainSizes[LoD][0]= SysTools::FromString<uint32_t>(elements[0]);
          m_DomainSizes[LoD][1]= SysTools::FromString<uint32_t>(elements[1]);
          m_DomainSizes[LoD][2]= SysTools::FromString<uint32_t>(elements[2]);
          if (elements[3] != "BrickCount") {
            std::cerr << "failed to parse line " << i << ": BrickCount not available" << std::endl;
            return false;
          }
          elements = SysTools::Tokenize(tokens[3], SysTools::PM_NONE);
          if (elements.size() != 3) {
            std::cerr << "failed to parse line " << i << ": invalid BrickCount" << std::endl;
            return false;
          }
          m_BrickCounts[LoD][0]= SysTools::FromString<uint32_t>(elements[0]);
          m_BrickCounts[LoD][1]= SysTools::FromString<uint32_t>(elements[1]);
          m_BrickCounts[LoD][2]= SysTools::FromString<uint32_t>(elements[2]);

        } else if (tokens[0] == " Subframe" || tokens[0] == "Subframe") {
          if (tokens.size() != 3) {
            std::cerr << "failed to parse line " << i << ": invalid Subframe" << std::endl;
            return false;
          }
          elements = SysTools::Tokenize(tokens[1], SysTools::PM_NONE);
          if (elements.size() != 2) {
            std::cerr << "failed to parse line " << i << ": invalid Subframe value" << std::endl;
            return false;
          }
          uint32_t const iSubframe = SysTools::FromString<uint32_t>(elements[0]);
          if (iSubframe != iSubframeCounter) {
            std::cerr << "failed to parse line " << i << ": wrong Subframe value" << std::endl;
            return false;
          }
          iExpectedBricks = SysTools::FromString<uint32_t>(tokens[2]);
          while (m_Frames.size() <= iFrameCounter)
            m_Frames.push_back(Frame());
          m_Frames.back().push_back(Subframe());

        } else if (tokens[0] == " Frame" || tokens[0] == "Frame") {
          if (tokens.size() != 4) {
            std::cerr << "failed to parse line " << i << ": invalid Frame" << std::endl;
            return false;
          }
          std::vector<std::string> elements;
          elements = SysTools::Tokenize(tokens[1], SysTools::PM_NONE);
          if (elements.size() != 2) {
            std::cerr << "failed to parse line " << i << ": invalid Frame value" << std::endl;
            return false;
          }
          uint32_t const iFrame = SysTools::FromString<uint32_t>(elements[0]);
          if (iFrame != iFrameCounter) {
            std::cerr << "failed to parse line " << i << ": wrong Frame value" << std::endl;
            return false;
          }
          while (m_Frames.size() <= iFrameCounter)
            m_Frames.push_back(Frame());
          iFrameCounter++;
          iExpectedBricks = 0;
          iSubframeCounter = 0;
        }
      }
    }
  }
  return true;
}

/** loads requests from a '.ba' file, in the format the HT wants. */
MALLOC unsigned*
requests_ba(const char* filename, size_t* n)
{
	*n = 0;
	BrickAccessFile baf(filename);

	if(!baf.Load()) {
		*n = 0;
		errno = EINVAL;
		return NULL;
	}
	const std::vector<BrickAccessFile::Frame>& frames = baf.GetFrames();
	for(const BrickAccessFile::Frame& f : frames) {
		for(const BrickAccessFile::Subframe& sf : f) {
			*n = *n + sf.size();
		}
	}
	if(verbose()) {
		fprintf(stderr, "Loading %zu bricks...\n", *n);
	}
	assert(*n > 0);
	unsigned* bricks = (unsigned*)malloc(sizeof(unsigned)*4*(*n));
	if(bricks == NULL) {
		*n = 0;
		errno = ENOMEM;
		return NULL;
	}
	size_t bid = 0; /* current brick ID, index into 'bricks' */
	for(size_t f=0; f < frames.size(); ++f) {
		for(size_t sf=0; sf < frames[f].size(); ++sf) {
			for(size_t brick=0; brick < frames[f][sf].size();
			    ++brick, ++bid) {
				bricks[bid*4+0] = frames[f][sf][brick][0];
				bricks[bid*4+1] = frames[f][sf][brick][1];
				bricks[bid*4+2] = frames[f][sf][brick][2];
				bricks[bid*4+3] = frames[f][sf][brick][3];
			}
		}
	}
	return bricks;
	return NULL;
}
