# Annotated Claims

## C001
- Text: Incident review confirmed two exposed admin endpoints.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [0:54] "Incident review confirmed two exposed admin endpoints."
- Bias Flags: none

## C002
- Text: The analyst report says this proves the platform will always fail under stress.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [55:134] "The analyst report says this proves the platform will always fail under stress."
- Bias Flags:
  - certainty: [109:115] always, [84:90] proves (fix: Use neutral phrasing for certainty language.)
