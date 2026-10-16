#include "speaker_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace fs = std::filesystem;

namespace diar {
namespace {

float Cosine(const std::vector<float>& a, const float* b, int dim) {
    double dot = 0.0, aa = 0.0, bb = 0.0;
    for (int i = 0; i < dim; ++i) {
        dot += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    return (float)(dot / (std::sqrt(aa * bb) + 1e-12));
}

// Speaker-cluster similarity: max cosine over the speaker's recent samples
// and its running mean.  Keeping the max over individual samples is more
// robust to drift than a single mean for long meetings.
float SpeakerSimilarity(const Speaker& speaker, const float* center, int dim) {
    float best = -1.0f;
    for (const auto& sample : speaker.samples) {
        if (sample.size() != static_cast<std::size_t>(dim)) continue;
        const float score = Cosine(sample, center, dim);
        if (score > best) best = score;
    }
    if (speaker.center.size() == static_cast<std::size_t>(dim)) {
        const float score = Cosine(speaker.center, center, dim);
        if (score > best) best = score;
    }
    return best;
}

// Compact O(n^3) Hungarian algorithm (rectangular: n rows <= m columns) on
// the cost matrix `cost` (row-major, n * m).  Returns for each row the
// assigned column, or -1.  With cluster counts <= 8 and speakers <= 20 this
// runs in microseconds.
std::vector<int> Hungarian(const std::vector<float>& cost, int n, int m) {
    const float kInf = 1e30f;
    // u/v potentials, p[j] = row assigned to column j, way[j] = previous
    // column on the alternating path (1-based textbook formulation).
    std::vector<float> u(n + 1, 0.0f), v(m + 1, 0.0f);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);
    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(m + 1, kInf);
        std::vector<char> used(m + 1, 0);
        do {
            used[j0] = 1;
            const int i0 = p[j0];
            float delta = kInf;
            int j1 = -1;
            for (int j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const float cur = cost[(i0 - 1) * m + (j - 1)] - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int j = 0; j <= m; ++j) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }
    std::vector<int> row_col(n, -1);
    for (int j = 1; j <= m; ++j)
        if (p[j] > 0 && p[j] <= n) row_col[p[j] - 1] = j - 1;
    return row_col;
}

}  // namespace

void SpeakerTable::Load(const std::string& path) {
    speakers_.clear();
    next_id_ = 1;
    std::ifstream in(path);
    if (!in) return;
    std::string line;
    while (std::getline(in, line)) {
        const auto p = line.find("\"speaker_id\"");
        if (p == std::string::npos) continue;
        const auto q = line.find('"', line.find(':', p) + 1);
        const auto r = line.find('"', q + 1);
        if (q == std::string::npos || r == std::string::npos) continue;
        Speaker s;
        s.id = line.substr(q + 1, r - q - 1);
        const auto c = line.find("\"sample_count\"");
        if (c != std::string::npos)
            s.count = std::atoll(line.c_str() + line.find(':', c) + 1);
        const auto seen = line.find("\"last_seen_time\"");
        if (seen != std::string::npos)
            s.last_seen = std::atoll(line.c_str() + line.find(':', seen) + 1);
        const auto a = line.find('[', line.find("\"embedding_center\""));
        const auto b = line.find(']', a);
        if (a != std::string::npos && b != std::string::npos) {
            std::stringstream values(line.substr(a + 1, b - a - 1));
            std::string token;
            while (std::getline(values, token, ',')) {
                try { s.center.push_back(std::stof(token)); } catch (...) {}
            }
        }
        speakers_.push_back(std::move(s));
    }
    for (auto& s : speakers_) {
        if (!s.center.empty()) s.samples.push_back(s.center);
    }
    for (const auto& s : speakers_) {
        const auto p = s.id.find_last_of('-');
        if (p != std::string::npos)
            next_id_ = std::max(next_id_, std::atoi(s.id.c_str() + p + 1) + 1);
    }
}

bool SpeakerTable::Save(const std::string& path) const {
    fs::path p(path);
    std::error_code ec;
    if (!p.parent_path().empty()) fs::create_directories(p.parent_path(), ec);
    const fs::path tmp = p.string() + ".tmp";
    std::ofstream out(tmp);
    if (!out) return false;
    out << "{\n  \"speakers\": [\n";
    for (std::size_t i = 0; i < speakers_.size(); ++i) {
        const auto& s = speakers_[i];
        out << "    {\"speaker_id\":\"" << s.id
            << "\",\"sample_count\":" << s.count
            << ",\"last_seen_time\":" << s.last_seen
            << ",\"embedding_center\":[";
        for (std::size_t j = 0; j < s.center.size(); ++j)
            out << (j ? "," : "") << std::setprecision(8) << s.center[j];
        out << "]}" << (i + 1 == speakers_.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    out.close();
    fs::rename(tmp, p, ec);
    if (ec) { fs::remove(tmp, ec); return false; }
    return true;
}

std::vector<std::string> SpeakerTable::Assign(
    const std::vector<float>& centers, int dim,
    const std::vector<long long>& weights, float threshold, long long now) {
    const int clusters = static_cast<int>(weights.size());
    std::vector<std::string> ids(clusters);
    if (clusters <= 0 || dim <= 0 ||
        centers.size() < static_cast<std::size_t>(clusters) * dim)
        return ids;

    std::vector<int> order(clusters);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return weights[a] > weights[b];
    });

    std::vector<char> taken(speakers_.size(), 0);
    constexpr std::size_t kMaxSamples = 8;
    for (int index : order) {
        const float* center = centers.data() + static_cast<std::size_t>(index) * dim;
        int best = -1;
        float best_score = threshold;
        // Match against every recent centroid of each speaker: a single
        // running mean is too brittle for short, noisy segments and causes
        // the same voice to spawn new speaker ids over a long meeting.
        for (std::size_t i = 0; i < speakers_.size(); ++i) {
            if (taken[i]) continue;
            const auto& samples = speakers_[i].samples;
            if (samples.empty()) {
                if (speakers_[i].center.size() != static_cast<std::size_t>(dim)) continue;
                const float score = Cosine(speakers_[i].center, center, dim);
                if (score >= best_score) { best_score = score; best = (int)i; }
                continue;
            }
            for (const auto& sample : samples) {
                if (sample.size() != static_cast<std::size_t>(dim)) continue;
                const float score = Cosine(sample, center, dim);
                if (score >= best_score) { best_score = score; best = (int)i; }
            }
        }
        const long long weight = std::max<long long>(1, weights[index]);
        std::vector<float> observation(center, center + dim);
        if (best < 0) {
            Speaker speaker;
            speaker.id = "speaker-" + std::to_string(next_id_++);
            speaker.center = observation;
            speaker.samples.push_back(observation);
            speaker.count = weight;
            speaker.last_seen = now;
            speakers_.push_back(std::move(speaker));
            best = static_cast<int>(speakers_.size()) - 1;
            taken.resize(speakers_.size(), 0);
            std::fprintf(stderr, "[diar] created new %s weight=%lld (no cosine >= %.2f)\n",
                         speakers_[best].id.c_str(), weight, threshold);
        } else {
            Speaker& speaker = speakers_[best];
            const long long previous = std::max<long long>(1, speaker.count);
            for (int d = 0; d < dim; ++d)
                speaker.center[d] = (speaker.center[d] * (float)previous +
                                     center[d] * (float)weight) /
                                    (float)(previous + weight);
            speaker.samples.push_back(std::move(observation));
            if (speaker.samples.size() > kMaxSamples) speaker.samples.erase(speaker.samples.begin());
            speaker.count = previous + weight;
            std::fprintf(stderr, "[diar] matched %s cosine=%.3f weight=%lld history=%lld\n",
                         speaker.id.c_str(), best_score, weight, previous);
        }
        speakers_[best].last_seen = now;
        taken[best] = 1;
        ids[index] = speakers_[best].id;
    }
    return ids;
}

std::vector<std::string> SpeakerTable::AssignOnline(
    const std::vector<float>& centers, int dim,
    const std::vector<long long>& weights,
    const OnlineParams& params, long long now) {
    const int clusters = static_cast<int>(weights.size());
    std::vector<std::string> ids(clusters);
    if (clusters <= 0 || dim <= 0 ||
        centers.size() < static_cast<std::size_t>(clusters) * dim)
        return ids;

    // Active clusters only: weight 0 means no speech was ever mapped to the
    // AHC centroid, so its embedding is meaningless.
    std::vector<int> active;
    for (int c = 0; c < clusters; ++c)
        if (weights[c] > 0) active.push_back(c);

    // Step 1: similarity matrix (active clusters x existing speakers).
    const int n = static_cast<int>(active.size());
    const int m = static_cast<int>(speakers_.size());
    std::vector<float> sim(n * m, -1.0f);
    for (int i = 0; i < n; ++i) {
        const float* center = centers.data() +
            static_cast<std::size_t>(active[i]) * dim;
        for (int j = 0; j < m; ++j)
            sim[i * m + j] = SpeakerSimilarity(speakers_[j], center, dim);
    }

    // Step 2: joint optimal assignment on 1 - similarity.  This is the diart
    // SpeakerMap: one cluster cannot steal a speaker that a better-matching
    // cluster already claimed (greedy's failure mode with 3+ speakers).
    std::vector<int> match(n, -1);  // active index -> speaker index
    if (n > 0 && m > 0) {
        // Pad with dummy columns so rows <= columns always holds (a chunk can
        // carry more clusters than the table has speakers early in a
        // meeting); dummy assignments mean "unmatched".
        const int cols = std::max(n, m);
        const float kDummy = 2.0f;  // worse than any 1 - similarity
        std::vector<float> cost(n * cols, kDummy);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < m; ++j)
                cost[i * cols + j] = 1.0f - sim[i * m + j];
        const std::vector<int> row_col = Hungarian(cost, n, cols);
        for (int i = 0; i < n; ++i) {
            const int col = row_col[i];
            // Reject dummy columns and pairs below the similarity threshold.
            match[i] = (col >= 0 && col < m && sim[i * m + col] >= params.threshold)
                           ? col : -1;
        }
    }

    const long long min_new = static_cast<long long>(
        params.min_new_sec * params.sample_rate);
    const long long min_update = static_cast<long long>(
        params.min_update_sec * params.sample_rate);
    constexpr std::size_t kMaxSamples = 8;

    for (int i = 0; i < n; ++i) {
        const int c = active[i];
        const float* center = centers.data() + static_cast<std::size_t>(c) * dim;
        const long long weight = weights[c];
        int best = match[i];

        if (best < 0) {
            // Unmatched: create a new speaker only with enough speech and a
            // free slot (rho_update / delta_new gates: "um"/"right" backchannel
            // fragments must never spawn phantom ids).
            if (weight >= min_new &&
                static_cast<int>(speakers_.size()) < params.max_speakers) {
                Speaker speaker;
                speaker.id = "speaker-" + std::to_string(next_id_++);
                speaker.center.assign(center, center + dim);
                speaker.samples.push_back(std::vector<float>(center, center + dim));
                speaker.count = weight;
                speaker.last_seen = now;
                speakers_.push_back(std::move(speaker));
                best = static_cast<int>(speakers_.size()) - 1;
                std::fprintf(stderr,
                             "[diar] online: created %s weight=%.1fs "
                             "(no match >= %.2f, %.1fs speech gate passed)\n",
                             speakers_[best].id.c_str(), weight / 16000.0,
                             params.threshold, params.min_new_sec);
            } else {
                // Table full or too little speech: identify-only against the
                // nearest existing speaker (diart's fallback), no update.
                float best_score = -1.0f;
                for (int j = 0; j < m; ++j) {
                    if (sim[i * m + j] > best_score) {
                        best_score = sim[i * m + j];
                        best = j;
                    }
                }
                if (best >= 0) {
                    speakers_[best].last_seen = now;
                    ids[c] = speakers_[best].id;
                    std::fprintf(stderr,
                                 "[diar] online: identify-only %s cosine=%.3f "
                                 "weight=%.1fs (create gated)\n",
                                 speakers_[best].id.c_str(), best_score,
                                 weight / 16000.0);
                    continue;
                }
                // No speaker exists at all: leave unlabeled.
                ids[c] = "";
                continue;
            }
        } else {
            // Matched above threshold: update the running mean only with
            // enough speech (tiny fragments cannot drift a centroid).
            if (weight >= min_update) {
                Speaker& speaker = speakers_[best];
                const long long previous = std::max<long long>(1, speaker.count);
                for (int d = 0; d < dim; ++d)
                    speaker.center[d] = (speaker.center[d] * (float)previous +
                                         center[d] * (float)weight) /
                                        (float)(previous + weight);
                speaker.samples.push_back(std::vector<float>(center, center + dim));
                if (speaker.samples.size() > kMaxSamples)
                    speaker.samples.erase(speaker.samples.begin());
                speaker.count = previous + weight;
            }
            speakers_[best].last_seen = now;
        }
        ids[c] = speakers_[best].id;
    }
    return ids;
}

}  // namespace diar
