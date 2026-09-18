#!/usr/bin/env bash
set -e
mkdir -p /tmp/essentia_models
cd /tmp/essentia_models
BASE=https://essentia.upf.edu/models

urls=(
  "$BASE/classifiers/danceability/danceability-musicnn-msd-2.pb"
  "$BASE/classifiers/mood_happy/mood_happy-musicnn-msd-2.pb"
  "$BASE/classifiers/mood_sad/mood_sad-musicnn-msd-2.pb"
  "$BASE/classifiers/mood_aggressive/mood_aggressive-musicnn-msd-2.pb"
  "$BASE/classifiers/mood_party/mood_party-musicnn-msd-2.pb"
  "$BASE/classifiers/mood_relaxed/mood_relaxed-musicnn-msd-2.pb"
)

for u in "${urls[@]}"; do
  echo "fetching $u"
  curl -sSL -o "$(basename "$u")" "$u" -w '  status=%{http_code} size=%{size_download}\n' || echo "FAILED: $u"
done

ls -la
