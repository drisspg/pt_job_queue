# Make PR

Create a draft GitHub PR for the fix with a structured body.

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

### 2. Lint and Auto-fix
Run the linter on changed files and auto-fix any issues before committing:
```bash
cd $JOB_DIR/pytorch
pip install lintrunner lintrunner-adapters 2>/dev/null
lintrunner init 2>/dev/null
lintrunner -a 2>&1 || true
```
If `lintrunner` is not available, at minimum run:
```bash
$WORKSPACE/.venv/bin/python -m ruff check --fix .
$WORKSPACE/.venv/bin/python -m ruff format .
```

### 3. Create Branch and Commit
```bash
cd $JOB_DIR/pytorch
git checkout -b ptq/$ISSUE_NUMBER
git add -A
git commit -m "Fix #$ISSUE_NUMBER: <short description of fix>

<one-paragraph explanation of root cause and what the fix does>"
```

### 4. Push
Try pushing directly to pytorch/pytorch first. If that fails (permissions), fork and push:
```bash
git push origin ptq/$ISSUE_NUMBER 2>&1 || {
  gh repo fork pytorch/pytorch --clone=false
  git remote add fork $(gh repo view --json sshUrl -q .sshUrl 2>/dev/null || echo "git@github.com:$(gh api user -q .login)/pytorch.git")
  git push fork ptq/$ISSUE_NUMBER
}
```

### 5. Build PR Body and Create PR
Read these files to construct the PR body:
- `$JOB_DIR/report.md` — summary of the fix
- `$JOB_DIR/worklog.md` — full investigation log
- `$JOB_DIR/fix.diff` — the diff

The PR body MUST follow this exact template structure:

    ## Summary
    <paste contents of report.md here>

    ## User prompt
    > <extract the user's prompt from the first "> **User:**" line in worklog.md>

    <details><summary>Worklog</summary>

    <paste contents of worklog.md here>

    </details>

    <details><summary>Diff</summary>

    ```diff
    <paste contents of fix.diff here>
    ```

    </details>

    🤖 Generated with [Claude Code](https://claude.com/claude-code)

Write this body to `/tmp/pr_body.md`, then create the PR:
```bash
gh pr create \
  --draft \
  --title "Fix #$ISSUE_NUMBER: <short description>" \
  --body-file /tmp/pr_body.md \
  --base main
```

### 6. Record Result
Append the PR URL to the worklog:
```markdown
## Pull Request
- PR: <url from gh pr create output>
- Status: Draft
```

### 7. Fallback
If `gh auth` fails or PR creation fails for any reason:
1. Save the PR body to `$JOB_DIR/pr_body.md`
2. Save the branch name and push instructions to the worklog
3. The user can create the PR manually later

Do NOT let a PR creation failure block the overall task — the fix itself is the primary deliverable.
