// Persistent cross-segment speaker identity store for the diarization worker.
//
// The JSON layout intentionally matches the previous CampPlus speaker table
// (speaker_id / sample_count / last_seen_time / embedding_center) so existing
// files keep loading and clean-history.sh needs no changes.  Embedding centers
// are 256-D raw community-1 embeddings and are matched with cosine similarity.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace diar {

struct Speaker {
    std::string id;
    std::vector<float> center;                  // running mean, persisted
    std::vector<std::vector<float>> samples;    // recent per-segment centroids
    long long count = 0;
    long long last_seen = 0;
};

class SpeakerTable {
 public:
    // Parses a previously saved table; a missing file leaves the table empty.
    void Load(const std::string& path);
    // Atomically writes the table (tmp file + rename).
    bool Save(const std::string& path) const;

    // Assigns a global id to each per-segment cluster. `centers` holds
    // clusters * dim floats; `weights` is the per-cluster sample count, used
    // to process dominant clusters first and to update running means.
    // Clusters without a match at or above `threshold` create new speaker-N
    // ids.  Returns one id per input cluster (empty string when unmatched and
    // no centroid is available).
    std::vector<std::string> Assign(const std::vector<float>& centers, int dim,
                                    const std::vector<long long>& weights,
                                    float threshold, long long now);

    // diart-style online assignment for the realtime "online" mode.  Instead
    // of greedy per-cluster matching it solves the whole batch jointly:
    //   1. build a similarity matrix cluster x speaker (max cosine over the
    //      speaker's recent samples and running mean),
    //   2. run a Hungarian assignment on 1 - similarity,
    //   3. accept pairs at or above `threshold`,
    //   4. unmatched clusters only create a new speaker when they carry at
    //      least `min_new_sec` seconds of speech and the table is not full,
    //      otherwise they fall back to the nearest existing speaker
    //      (identify-only, no centroid update),
    //   5. matched clusters update the running mean only when they carry at
    //      least `min_update_sec` seconds of speech.
    struct OnlineParams {
        float threshold = 0.55f;
        float min_new_sec = 2.0f;     // speech needed before creating an id
        float min_update_sec = 1.0f;  // speech needed to move a centroid
        int max_speakers = 20;
        int sample_rate = 16000;
    };
    std::vector<std::string> AssignOnline(
        const std::vector<float>& centers, int dim,
        const std::vector<long long>& weights,
        const OnlineParams& params, long long now);

    std::size_t size() const { return speakers_.size(); }
    const std::vector<Speaker>& speakers() const { return speakers_; }
    // Drops all speakers (session boundary), restarting id allocation at 1.
    void Clear() { speakers_.clear(); next_id_ = 1; }

 private:
    std::vector<Speaker> speakers_;
    int next_id_ = 1;
};

}  // namespace diar
