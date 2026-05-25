#!/usr/bin/env bash
set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
DOCKERFILE="$REPO_ROOT/usecase/Dockerfile"

SHA="$(git rev-parse HEAD)"

sed -i "s|git+https://github.com/PardalZ7/sentinel.git@[a-f0-9]*#|git+https://github.com/PardalZ7/sentinel.git@${SHA}#|" "$DOCKERFILE"

echo "Dockerfile atualizado para SHA: $SHA"

git add "$DOCKERFILE"
git commit -m "chore: pin sentinel SHA to $SHA"
git push
