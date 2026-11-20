# Unattended run (started 20 November 2026, 16:54)

The remaining passes run without any Claude or VS Code session. They were
started through Windows' process-creation service, so they are not children of
the editor; closing the editor does not stop them.

## What is running

`tools/run_unattended.sh`, inside the `XCResearch` Linux VM. All numerics on
the GPU, three host cores with one thread each, all data on the D drive.

| Core | Job |
|---|---|
| 0 | Benchmark PBE run (`pbe_qzvp_g3`, all 2,442 structures) and semilocal pass |
| 1 | Benchmark response pass, every size, including the blocked path for the largest |
| 2 | Outside reaction set (ACCDB) in size tiers up to 500 basis functions, then status reports |

Every pass skips structures that already have a result, so it can be stopped
and restarted at any time.

## Where to look

- **`D:\xcresearch\status\latest.txt`**: rewritten every 2 hours. Coverage
  counts, error count, benchmark-only fits, and the train-outside,
  test-on-benchmark numbers. A final version with the official pinned scorer
  is written when all passes finish.
- `D:\xcresearch\logs\unattended.log`: start, restarts, finish.
- `D:\xcresearch\logs\*.log`: one line per structure for each pass.

## Conditions

- The machine must stay awake. Sleep or hibernation pauses everything; a
  reboot stops it (restart with the command below).
- No one is watching for problems. A structure whose process dies twice (for
  example out of memory on the 8 GB GPU or the 6 GB VM) is recorded as an error
  and skipped, so the run does not stall on it. Those structures need a decision
  later; until they are redone there is no full-suite score.
- The large-structure path (above roughly 1,700 basis functions) was validated
  for correctness on a small molecule but has not yet run at full size.

## Stop

    wsl.exe --terminate XCResearch

## Start again (from PowerShell, survives closing the terminal)

    Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = 'wsl.exe -d XCResearch -u root --exec bash /mnt/c/Users/breth/myrepo/electronfield-killtest/tools/run_unattended.sh' }

A second copy refuses to start while one is active.

## Estimated duration

Two to three days for the whole benchmark at the measured rate (about two
structures per minute at 700 basis functions, slowing with size). This is an
estimate from measured scaling, not a schedule.

## When it finishes, or when a session resumes

1. Read `latest.txt`.
2. List result files containing `"error"` and decide whether to retry them
   (move the file aside and rerun the pass).
3. `python -m xcgpu.fit`, `xcgpu.fit_general` and `xcgpu.score_model` reproduce
   every number in the status file.
