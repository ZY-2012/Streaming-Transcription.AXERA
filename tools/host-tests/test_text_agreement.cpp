// Host-side tests (no NPU) for the streaming ASR track's pure text logic:
// punctuation-loose LocalAgreement-2 (moved verbatim from StreamSlicer, so
// these also guard the offline path) plus the char<->time mapping and turn
// splitting used to attribute one epoch's text across speaker changes.
//
//   g++ -std=c++17 -O2 -I framework tools/host-tests/test_text_agreement.cpp \
//       -o build_host/test_text_agreement
#include "text_agreement.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace textutil;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", what);
  }
}

void TestAgreement() {
  // Identical content with different punctuation agrees in full and keeps
  // the newer hypothesis's punctuation form.
  check(CommonPrefixLoose("今天天气不错吧", "今天天气不错吧？") ==
        std::string("今天天气不错吧").size(), "loose prefix keeps new punct");
  check(StripPunct("好，谢谢！") == "好谢谢", "strip punct");
  check(CommonPrefixLoose("abc", "abd") == 2, "divergence at c/d");

  // Punctuation-only difference inside the prefix does not stall the merge.
  // ("哎，小队" vs "哎小帅听得到": the punct in a is skipped and 小 still
  // matches, so the agreeing region in b is "哎小" = 6 bytes.)
  check(CommonPrefixLoose("哎，小队", "哎小帅听得到") == 6, "punct skip in prefix");

  // ContentMatchOffset locates the confirmed content inside a hypothesis
  // whose punctuation moved.
  const std::string confirmed = StripPunct("今天天气不错吧");
  check(ContentMatchOffset("今天，天气不错吧？", confirmed) ==
        std::string("今天，天气不错吧").size(), "content match offset");
  check(ContentMatchOffset("完全不同", confirmed) == 0, "no match offset 0");
}

void TestOffsetsAndSplit() {
  const std::string s = "好的，谢谢！再见";
  const auto offsets = ContentOffsets(s);
  check(offsets.size() == 6, "content char count");
  const auto pieces = SplitAt(s, {3});
  check(pieces.size() == 2, "split in two");
  // Content index 3 = the first three content chars (好的谢); the ！ stays
  // with the following piece.
  check(pieces[0] == "好的，谢" && pieces[1] == "谢！再见", "split pieces");

  // Boundary at 0 / total is ignored (single piece).
  check(SplitAt(s, {0, 6}).size() == 1, "edge indices ignored");

  // Multiple cuts.
  const auto three = SplitAt("甲乙丙丁戊己", {2, 4});
  check(three.size() == 3 && three[0] == "甲乙" && three[1] == "丙丁" &&
        three[2] == "戊己", "three-way split");
}

void TestSnapToPunct() {
  const std::string s = "好，谢谢。好，非常感谢";
  const auto offsets = ContentOffsets(s);
  // content: 好谢谢好非常感谢 = 8 chars (indices 0..7).  Punct gaps sit
  // before index 3 (。after 谢谢) and index 4 (，after the second 好).  A
  // raw cut at 7 (late by ~4 chars, the http poll cadence) snaps to the
  // NEAREST punct, index 4 - "好，谢谢。好，|非常感谢"; the alternative at 3
  // is one char further.  One boundary word can swap sides at this poll
  // granularity; the worker backend's 100 ms partials tighten it.
  check(SnapToPunct(s, offsets, 7, 5) == 4, "snap back to nearest punct");
  // Forward preference on ties: a raw cut at 2 snaps forward to 3, keeping
  // "好，谢谢" together instead of splitting after the leading 好，.
  check(SnapToPunct(s, offsets, 2, 5) == 3, "snap forward preferred");
  // Strong punctuation wins over a comma sitting exactly on the estimate:
  // the cut goes after "好，谢谢。" so speaker_1 keeps its opening word.
  check(SnapToPunct(s, offsets, 4, 5) == 3, "strong punct beats adjacent comma");
  // No punctuation anywhere near -> unchanged.
  const std::string t = "甲乙丙丁";
  check(SnapToPunct(t, ContentOffsets(t), 2, 3) == 2, "no punct, no snap");
}

void TestCharsAt() {
  std::vector<CharTimeAnchor> anchors = {{0, 1600}, {4, 3200}, {8, 6400}};
  check(CharsAt(anchors, 800) == 0, "before first anchor");
  check(CharsAt(anchors, 6400) == 8, "at last anchor");
  check(CharsAt(anchors, 99999) == 8, "after last anchor");
  // Between anchor 0 (1600, 0 chars) and 1 (3200, 4 chars): midpoint 2400
  // -> 2 chars.
  check(CharsAt(anchors, 2400) == 2, "linear interpolation");
  // Between anchors 1 (3200, 4) and 2 (6400, 8): 4800 -> 6.
  check(CharsAt(anchors, 4800) == 6, "second interval");
  check(CharsAt({}, 12345) == 0, "empty anchors");
}

void TestCharsAtOrigin() {
  // Head extrapolation: 0 chars at the epoch origin, first anchor 5@40448
  // (the 2P opening epoch b=27904).  A turn boundary before the first
  // anchor must interpolate instead of clamping to 0.
  std::vector<CharTimeAnchor> anchors{{5, 40448}, {8, 56576}};
  check(CharsAt(anchors, 27904, 27904) == 0, "at origin");
  check(CharsAt(anchors, 26000, 27904) == 0, "before origin");
  // (40000-27904)/(40448-27904) * 5 = 4.83 -> 4
  check(CharsAt(anchors, 40000, 27904) == 4, "head extrapolation");
  check(CharsAt(anchors, 40448, 27904) == 5, "at first anchor");
  // above the first anchor behaves like the plain map
  check(CharsAt(anchors, 48000, 27904) == 6, "past first anchor");
  check(CharsAt(anchors, 99999, 27904) == 8, "clamped at last");
}

}  // namespace

int main() {
  TestAgreement();
  TestOffsetsAndSplit();
  TestSnapToPunct();
  TestCharsAt();
  TestCharsAtOrigin();
  if (failures == 0) {
    std::printf("text agreement: OK\n");
    return 0;
  }
  std::printf("text agreement: %d FAILURES\n", failures);
  return 1;
}
