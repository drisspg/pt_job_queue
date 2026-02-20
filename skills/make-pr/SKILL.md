# Make PR

Create a draft GitHub PR for the fix.

## Prerequisites
- fix.diff is non-empty
- Repro script passes (bug is fixed)
- Verification checks have been run (use verify-fix skill first)

## Steps

### 1. Validate
Confirm fix.diff exists and is non-empty:
```bash
test -s $JOB_DIR/fix.diff || echo "ERROR: fix.diff is empty or missing"
```

### 2. Create Branch and Commit
```bash
cd $JOB_DIR/pytorch
git checkout -b ptq/$ISSUE_NUMBER
git add -A
git commit -m "Fix #$ISSUE_NUMBER: <short description of fix>

<one-paragraph explanation of root cause and what the fix does>"
```

### 3. Push and Create PR
Try pushing directly to pytorch/pytorch first. If that fails (permissions), fork and push:
```bash
# Try direct push
git push origin ptq/$ISSUE_NUMBER 2>&1 || {
  # Fork and push
  gh repo fork pytorch/pytorch --clone=false
  git remote add fork $(gh repo view --json sshUrl -q .sshUrl 2>/dev/null || echo "git@github.com:$(gh api user -q .login)/pytorch.git")
  git push fork ptq/$ISSUE_NUMBER
}
```

Create the draft PR using the report as the body:
```bash
gh pr create \
  --draft \
  --title "Fix #$ISSUE_NUMBER: <short description>" \
  --body-file $JOB_DIR/report.md \
  --base main
```

### 4. Record Result
Append the PR URL to the worklog:
```markdown
## Pull Request
- PR: <url from gh pr create output>
- Status: Draft
```

### 5. Fallback
If `gh auth` fails or PR creation fails for any reason:
1. Save the PR body to `$JOB_DIR/pr_body.md`
2. Save the branch name and push instructions to the worklog
3. The user can create the PR manually later

Do NOT let a PR creation failure block the overall task — the fix itself is the primary deliverable.
