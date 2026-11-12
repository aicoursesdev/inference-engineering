#!/usr/bin/env bash
# Usage: ./triage.sh <owner/repo> <issue number>
# Is this "good first issue" still free? Needs the GitHub CLI (gh), logged in.
set -euo pipefail
repo=$1; n=$2
echo "== the issue"
gh issue view "$n" -R "$repo" --json title,state,assignees,labels,updatedAt \
  --jq '"\(.state)  \(.title)\nassigned to: \([.assignees[].login] | join(", "))\nlast update: \(.updatedAt)"'
echo "== pull requests that mention it (any state)"
gh pr list -R "$repo" -S "$n" --state all --json number,state,title \
  --jq '.[] | "#\(.number) \(.state)  \(.title)"'
echo "== open good first issues in this repo"
gh issue list -R "$repo" -l "good first issue" -s open -L 200 --json number --jq 'length'
