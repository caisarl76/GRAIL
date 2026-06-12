# Real Robot Validation Notes

This page records concise, non-sensitive operator validation notes for real
Unitree G1 tests. Do not store robot IPs, private network details, tokens, or
site-specific safety information here.

## 2026-06-12 Ramp Test

Context:

- Robot: Unitree G1.
- Test type: real robot deployment validation.
- Feature under test: ramp behavior.
- Operator result: ramp feature works well.

Notes:

- Preserve the exact deploy command, checkpoint path, and ramp parameters in
  the local run log for the deployment session.
- Keep simulation and dry-run checks as the gate before repeating real robot
  tests.
- Do not promote unrelated pick-and-place policies to real robot deployment
  based on this ramp result; those policies still need separate simulation
  success criteria.

