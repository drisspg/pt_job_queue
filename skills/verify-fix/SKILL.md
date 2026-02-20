# Verify Fix

Run tests and performance checks after applying a fix, before creating a PR.

## Steps

### 1. Repro Check
Re-run the repro script to confirm the bug is fixed:
```bash
$WORKSPACE/.venv/bin/python $JOB_DIR/repro.py
```
If the repro still fails, stop and report — the fix is incomplete.

### 2. Regression Tests
Find and run related PyTorch tests. Look for test files that exercise the same module as the fix:
```bash
# Find relevant test files based on the changed modules
cd $JOB_DIR/pytorch
git diff --name-only | while read f; do
  module=$(echo "$f" | sed 's|/|.|g' | sed 's|\.py$||')
  echo "Changed: $module"
done

# Run targeted tests
$WORKSPACE/.venv/bin/python -m pytest test/ -x -q -k "relevant_test_name" 2>&1 | head -50
```
If tests fail, note which ones and whether they are pre-existing failures or regressions from the fix.

### 3. Performance Check
Run a basic before/after comparison using the repro script:
```bash
# Time with fix applied
time $WORKSPACE/.venv/bin/python $JOB_DIR/repro.py

# Time without fix (stash changes, re-apply site-packages from clean state)
cd $JOB_DIR/pytorch && git stash
bash $WORKSPACE/scripts/apply_to_site_pkgs.sh $JOB_DIR/pytorch
time $WORKSPACE/.venv/bin/python $JOB_DIR/repro.py

# Restore fix
cd $JOB_DIR/pytorch && git stash pop
bash $WORKSPACE/scripts/apply_to_site_pkgs.sh $JOB_DIR/pytorch
```
Report any significant performance regressions (>10% slowdown).

### 4. Summary
Append verification results to the worklog:
```markdown
## Verification Results
- Repro check: PASS/FAIL
- Regression tests: X passed, Y failed (list failures)
- Performance: No regression / N% slower (details)
```
