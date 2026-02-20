# Fix Review

Address PR review comments and push fixes.

## Prerequisites
- A draft PR has been created (check worklog for the PR URL)
- The PR has review comments to address

## Steps

### 1. Get the PR Number
Extract the PR number from the worklog's "Pull Request" section, or from the user's prompt.

### 2. Fetch Review Comments
```bash
gh pr view $PR_NUMBER --repo pytorch/pytorch --json reviews,comments --jq '.reviews[].body, .comments[].body'
```

For inline review comments (the most common):
```bash
gh api repos/pytorch/pytorch/pulls/$PR_NUMBER/comments --jq '.[] | "File: \(.path):\(.line // .original_line)\nComment: \(.body)\n---"'
```

### 3. Address Each Comment
For each review comment:
1. Read the referenced file and line
2. Understand what the reviewer is asking for
3. Make the requested change (or explain in the worklog why you disagree)
4. Update the worklog with what you changed

### 4. Lint and Auto-fix
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

### 5. Test
Re-run the repro script and any additional tests to verify the fix still works:
```bash
$WORKSPACE/.venv/bin/python $JOB_DIR/repro.py
```

### 6. Update Diff and Report
```bash
cd $JOB_DIR/pytorch && git diff > $JOB_DIR/fix.diff
```
Update `report.md` with a section noting what review feedback was addressed.

### 7. Commit and Push
```bash
cd $JOB_DIR/pytorch
git add -A
git commit -m "Address review feedback on Fix #$ISSUE_NUMBER

<summary of changes made in response to review>"
git push
```

### 8. Reply to Reviews (optional)
If `gh` auth is working, you can reply to review comments:
```bash
gh api repos/pytorch/pytorch/pulls/$PR_NUMBER/comments/$COMMENT_ID/replies \
  -f body="Done — <brief description of change>"
```

### 9. Record in Worklog
Append to the worklog:
```markdown
## Review Fixes
- Addressed N review comments
- Changes: <summary>
- Tests: still passing
- Pushed to branch ptq/$ISSUE_NUMBER
```
