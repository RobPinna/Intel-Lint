# Annotated Claims

## C001
- Text: Threat monitoring identified repeated login attempts against the admin console.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [0:79] "Threat monitoring identified repeated login attempts against the admin console."
- Bias Flags: none

## C002
- Text: The briefing describes a breakthrough detection model that never misses phishing attempts.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [80:170] "The briefing describes a breakthrough detection model that never misses phishing attempts."
- Bias Flags:
  - certainty: [139:144] never (fix: Use neutral phrasing for certainty language.)
  - hype: [105:117] breakthrough (fix: Use neutral phrasing for hype language.)

## C003
- Text: No independent telemetry was provided for the attribution statement.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [171:239] "No independent telemetry was provided for the attribution statement."
- Bias Flags: none
