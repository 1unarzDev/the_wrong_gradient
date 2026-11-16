# Analysis plan, revision round (written 2026-11-15T18:47:26-06:00, before any of these results were viewed)

All runs reuse the replication's fresh courses (seeds 2000–2199) unless stated; paired by course; sensor perception.

1. New strategies, 4 and 2 boats, both radios (data/v2_new_policies_n4, data/v2_new_policies_n2):
   - P5: information-limited central assignment (leader boat plans on its own map and radio-received positions;
     assignments delivered only by radio).
   - R3: reservation-based marginal contribution (claims trusted until the claimant's announced completion time plus
     a margin, then decay).
   Planned contrasts (Holm within each fleet size, task points; total points reported alongside):
     R3 − P3 (degraded), R3 − A2 (degraded), P4 − P5 (degraded), P5 − P3 (degraded).
2. Robustness (4 boats, unreliable radio, seeds 3100–3199): harsher radio (p_good_to_bad 0.25, drop_good 0.3,
   latency up to 5 s) and slower boats (v_max 1.6). Contrasts P4 − P3, A2 − P3, R3 − P3 (task points), reported with
   unadjusted 95% intervals; descriptive check of whether the sign and rough size persist.
3. Single boat (competition configuration), seeds 4000–4199, 600 s episodes:
   graph ETA (P2) vs control-aware ETA (A2); A2 with bonus-aware values off; A2 with bay inference off; greedy (P0).
   Contrasts (Holm over 3): A2 − P2, A2 − A2[no bonus], A2 − A2[no bay inference]; task points and total points.

## Amendment (2026-11-15T18:56:33-06:00, before any round-3 result was viewed)
Item 3 (single-boat study) is withdrawn: the 2026 rules allow up to two vehicles, so the competition-relevant
configuration is the two-boat fleet. A single boat is run only as a fleet-size comparison point: P2 and A2, one boat,
same 200 courses (seeds 2000–2199), same 300 s episodes, unreliable radio (data/v2_one_boat). Descriptive only.

## Exploratory follow-up (2026-11-15T19:19:11-06:00, AFTER viewing the 4-boat R3/P5 results)
R3 did not reduce duplication relative to P3. Diagnostic (A2, 20 courses): announced claim arrivals are times to
reach a task, not to finish it; 97% of speed-challenge claims finish >5 s after the announced arrival (median 22.7 s).
New exploratory strategy L3 (heartbeat lease: a claim is valid while the claimant keeps broadcasting it; decays only
after >8 s of silence). Runs: 4 and 2 boats, both radios, seeds 2000–2199 (data/v2_lease_n4, data/v2_lease_n2).
Contrasts L3 − P3 and L3 − A2 (task points), reported as exploratory (unadjusted CIs, Holm over these 2 within each
fleet size).
