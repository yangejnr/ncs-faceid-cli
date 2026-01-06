#!/usr/bin/env bash
set -euo pipefail

# Push Docker image to ECR with:
#  - latest tag
#  - immutable tag (defaults to git SHA; can be overridden)
#
# Usage:
#   ./scripts/push_to_ecr_latest_and_version.sh <aws-region> <ecr-repo-name> [version_tag] [profile]
#
# Examples:
#   ./scripts/push_to_ecr_latest_and_version.sh eu-west-2 ncs-faceid-cli
#   ./scripts/push_to_ecr_latest_and_version.sh eu-west-2 ncs-faceid-cli 0.2.0
#   ./scripts/push_to_ecr_latest_and_version.sh eu-west-2 ncs-faceid-cli 0.2.0 myprofile

AWS_REGION="${1:-}"
REPO_NAME="${2:-}"
VERSION_TAG="${3:-}"
PROFILE="${4:-}"

if [[ -z "$AWS_REGION" || -z "$REPO_NAME" ]]; then
  echo "Usage: $0 <aws-region> <ecr-repo-name> [version_tag] [profile]"
  exit 1
fi

aws_cmd=(aws)
if [[ -n "$PROFILE" ]]; then
  aws_cmd+=(--profile "$PROFILE")
fi

command -v docker >/dev/null 2>&1 || { echo "docker not found"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git not found"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found"; exit 1; }

if [[ -z "$VERSION_TAG" ]]; then
  VERSION_TAG="$(git rev-parse --short HEAD)"
fi

AWS_ACCOUNT_ID="$("${aws_cmd[@]}" sts get-caller-identity --query Account --output text)"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

IMAGE_LOCAL="${REPO_NAME}:build"
IMAGE_LATEST_REMOTE="${ECR_URI}/${REPO_NAME}:latest"
IMAGE_VERSION_REMOTE="${ECR_URI}/${REPO_NAME}:${VERSION_TAG}"

echo "Account:     ${AWS_ACCOUNT_ID}"
echo "Region:      ${AWS_REGION}"
echo "Repository:  ${REPO_NAME}"
echo "ECR URI:     ${ECR_URI}"
echo "Tags:        latest, ${VERSION_TAG}"
echo

echo "[1/7] Ensure ECR repository exists..."
"${aws_cmd[@]}" ecr describe-repositories --repository-names "${REPO_NAME}" --region "${AWS_REGION}" >/dev/null 2>&1 \
  || "${aws_cmd[@]}" ecr create-repository --repository-name "${REPO_NAME}" --region "${AWS_REGION}" >/dev/null

echo "[2/7] Login Docker to ECR..."
"${aws_cmd[@]}" ecr get-login-password --region "${AWS_REGION}" | docker login --username AWS --password-stdin "${ECR_URI}" >/dev/null

echo "[3/7] Build image..."
docker build -t "${IMAGE_LOCAL}" .

echo "[4/7] Tag image..."
docker tag "${IMAGE_LOCAL}" "${IMAGE_LATEST_REMOTE}"
docker tag "${IMAGE_LOCAL}" "${IMAGE_VERSION_REMOTE}"

echo "[5/7] Push latest..."
docker push "${IMAGE_LATEST_REMOTE}"

echo "[6/7] Push version tag..."
docker push "${IMAGE_VERSION_REMOTE}"

echo "[7/7] Verify most recent images..."
"${aws_cmd[@]}" ecr describe-images \
  --repository-name "${REPO_NAME}" \
  --region "${AWS_REGION}" \
  --query "sort_by(imageDetails,& imagePushedAt)[-5:].{pushed:imagePushedAt,tags:imageTags}" \
  --output table

echo
echo "Pushed:"
echo "  ${IMAGE_LATEST_REMOTE}"
echo "  ${IMAGE_VERSION_REMOTE}"
