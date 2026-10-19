#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Punctuation-loose Local-Agreement-2 text helpers, shared by the offline
// StreamSlicer (growing-window re-transcription) and the StreamingAsrTrack
// (continuous streaming partials).  SenseVoice jitters punctuation across
// overlapping windows ("吧？" -> "吧" -> "吧？"), which stalls a naive
// byte-wise common prefix right after an epoch opening; the merge therefore
// compares CONTENT (punctuation skipped) and commits the newer hypothesis's
// own bytes (punctuation included) up to the last stable content character.
//
// The streaming-track additions below add a coarse char<->time mapping: the
// track records, at every confirmation, (content chars confirmed, absolute
// audio position).  At epoch close the diarization timeline's turn
// boundaries are projected onto the text through that mapping so one
// epoch's text can be split at speaker changes.
namespace textutil {

inline size_t Utf8Len(unsigned char c) {
  if (c < 0x80) return 1;
  if ((c & 0xE0) == 0xC0) return 2;
  if ((c & 0xF0) == 0xE0) return 3;
  if ((c & 0xF8) == 0xF0) return 4;
  return 1;
}

inline bool IsPunct(const std::string& s, size_t i, size_t len) {
  const auto c = static_cast<unsigned char>(s[i]);
  if (len == 1) {
    return c == ',' || c == '.' || c == '?' || c == '!' || c == ';' ||
           c == ':' || c == '"' || c == '\'' || c == '(' || c == ')' ||
           c == '[' || c == ']' || c == '{' || c == '}' || c == '-' ||
           c == '/' || c == '|';
  }
  uint32_t cp = 0;
  if (len == 2) {
    cp = ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
  } else if (len == 3) {
    cp = ((c & 0x0Fu) << 12) |
         ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
         (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
  } else {
    return false;
  }
  return (cp >= 0x3001 && cp <= 0x3002) ||   // 、。
         (cp >= 0x3008 && cp <= 0x301F) ||   // 〈〉《》「」『』【】…
         (cp >= 0xFF01 && cp <= 0xFF65) ||   // ，？！：；（）—— fullwidth
         (cp >= 0x2018 && cp <= 0x201F) ||   // ' ' " "
         cp == 0x2026 || cp == 0x2014;      // … —
}

inline std::string StripPunct(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    const size_t l = Utf8Len(static_cast<unsigned char>(s[i]));
    if (!IsPunct(s, i, l)) out.append(s, i, l);
    i += l;
  }
  return out;
}

// Byte length of the prefix of `b` covering the content both hypotheses
// agree on; punctuation inside the agreeing region keeps `b`'s own form.
inline size_t CommonPrefixLoose(const std::string& a, const std::string& b) {
  size_t ia = 0, ib = 0, disp = 0;
  while (ia < a.size() && ib < b.size()) {
    const size_t la = Utf8Len(static_cast<unsigned char>(a[ia]));
    const size_t lb = Utf8Len(static_cast<unsigned char>(b[ib]));
    if (la == lb && a.compare(ia, la, b, ib, lb) == 0) {
      ia += la; ib += lb; disp = ib;   // identical char (content or punct)
      continue;
    }
    const bool pa = IsPunct(a, ia, la), pb = IsPunct(b, ib, lb);
    if (pa && pb) { ia += la; ib += lb; continue; }  // both punct: agree to skip
    if (pa) { ia += la; continue; }
    if (pb) { ib += lb; continue; }
    break;                                          // real divergence
  }
  return disp;
}

// Byte offset in `hyp` after its content prefix matches `prefix` (already
// punctuation-stripped); stops at the first content mismatch.  Used to
// split a hypothesis into (confirmed display text, provisional tail).
inline size_t ContentMatchOffset(const std::string& hyp,
                                 const std::string& prefix) {
  size_t i = 0, j = 0;
  while (j < prefix.size() && i < hyp.size()) {
    const size_t l = Utf8Len(static_cast<unsigned char>(hyp[i]));
    if (IsPunct(hyp, i, l)) { i += l; continue; }
    if (hyp.compare(i, l, prefix, j, l) != 0) break;
    i += l; j += l;
  }
  return i;
}

// ---------------- streaming-track additions ----------------

// Byte offsets where each content (non-punctuation) character of `s`
// starts; ContentOffsets(s).size() == StripPunct(s).size().
inline std::vector<size_t> ContentOffsets(const std::string& s) {
  std::vector<size_t> out;
  size_t i = 0;
  while (i < s.size()) {
    const size_t l = Utf8Len(static_cast<unsigned char>(s[i]));
    if (!IsPunct(s, i, l)) out.push_back(i);
    i += l;
  }
  return out;
}

// One confirmation point: `chars` content characters had been confirmed
// once the session had consumed up to absolute audio position `pos`
// (16 kHz samples since stream start).  Anchors are kept sorted by pos
// with strictly increasing positions (equal-pos updates replace).
struct CharTimeAnchor {
  size_t chars = 0;
  size_t pos = 0;
};

// Content-char count corresponding to absolute audio position `t`:
// 0 before the first anchor, the last anchor's count at/after it, and a
// linear interpolation in between.  Coarse (one point per confirmation)
// but enough to project diarization turn boundaries onto text.
inline size_t CharsAt(const std::vector<CharTimeAnchor>& anchors, size_t t) {
  if (anchors.empty()) return 0;
  if (t <= anchors.front().pos) return 0;
  if (t >= anchors.back().pos) return anchors.back().chars;
  size_t lo = 0, hi = anchors.size() - 1;
  while (lo + 1 < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (anchors[mid].pos <= t) lo = mid; else hi = mid;
  }
  const CharTimeAnchor& a = anchors[lo];
  const CharTimeAnchor& b = anchors[hi];
  if (b.pos <= a.pos) return a.chars;
  const double frac = static_cast<double>(t - a.pos) /
                      static_cast<double>(b.pos - a.pos);
  return a.chars + static_cast<size_t>(frac *
                                      static_cast<double>(b.chars - a.chars));
}

// Split `s` at the given sorted content-char indices (each index is the
// first character of the next piece; 0 and the total are ignored).
inline std::vector<std::string> SplitAt(const std::string& s,
                                        const std::vector<size_t>& content_indices) {
  std::vector<std::string> out;
  const std::vector<size_t> offsets = ContentOffsets(s);
  std::vector<size_t> byte_cuts;
  byte_cuts.push_back(0);
  for (size_t idx : content_indices) {
    if (idx == 0 || idx >= offsets.size()) continue;
    byte_cuts.push_back(offsets[idx]);
  }
  byte_cuts.push_back(s.size());
  std::sort(byte_cuts.begin(), byte_cuts.end());
  byte_cuts.erase(std::unique(byte_cuts.begin(), byte_cuts.end()),
                  byte_cuts.end());
  for (size_t i = 0; i + 1 < byte_cuts.size(); ++i) {
    if (byte_cuts[i] < byte_cuts[i + 1])
      out.push_back(s.substr(byte_cuts[i], byte_cuts[i + 1] - byte_cuts[i]));
  }
  return out;
}

// Snap a content-char cut index to the nearest punctuation boundary within
// +-window content chars ("好的，谢谢好非常" cut at 7 snaps to the 。after
// 谢谢).  Cuts land between words instead of mid-word when the coarse
// char<->time interpolation is off by a character or two.
inline std::size_t SnapToPunct(const std::string& s, const std::vector<std::size_t>& offsets,
                               std::size_t idx, std::size_t window) {
  if (offsets.empty() || idx == 0 || idx >= offsets.size()) return idx;
  auto gap_is_punct = [&](std::size_t j) {
    if (j == 0) return false;
    const std::size_t begin =
        offsets[j - 1] + Utf8Len(static_cast<unsigned char>(s[offsets[j - 1]]));
    const std::size_t end = offsets[j];
    for (std::size_t p = begin; p < end; p += Utf8Len(static_cast<unsigned char>(s[p])))
      if (IsPunct(s, p, Utf8Len(static_cast<unsigned char>(s[p]))) ) return true;
    return false;
  };
  // Prefer the forward candidate on ties: boundary estimates tend to sit
  // slightly late relative to the true speaker switch, so the next phrase
  // end forward is the likelier cut.
  for (std::size_t d = 0; d <= window; ++d) {
    if (idx + d < offsets.size() && gap_is_punct(idx + d)) return idx + d;
    if (idx >= d && idx - d > 0 && gap_is_punct(idx - d)) return idx - d;
  }
  return idx;
}

}  // namespace textutil
