# Examples

## Scenario 1: Minimal incident summary
Input: `examples/scenario_minimal/input.txt`

Run:
```bash
py -3 -m intel_lint.cli.main examples/scenario_minimal/input.txt --out examples/scenario_minimal/out
```

Expected snippet:
```text
claims=2 engine=placeholder
```

## Scenario 2: Certainty-language bias
Input: `examples/scenario_bias/input.txt`

Run:
```bash
py -3 -m intel_lint.cli.main examples/scenario_bias/input.txt --out examples/scenario_bias/out
```

Expected snippet:
```json
"tag": "certainty"
```

## Sample dataset
- `examples/sample_data/manifest.jsonl`
- Sanitized synthetic records only.
