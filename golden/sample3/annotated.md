# Annotated Claims

## C001
- Text: A weekly CTI summary reported suspicious MFA fatigue alerts in one business unit.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [0:81] "A weekly CTI summary reported suspicious MFA fatigue alerts in one business unit."
- Bias Flags: none

## C002
- Text: The note says it is undeniable that the campaign guarantees compromise across all regions.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [82:172] "The note says it is undeniable that the campaign guarantees compromise across all regions."
- Bias Flags:
  - certainty: [102:112] undeniable (fix: Use neutral phrasing for certainty language.)

## C003
- Text: Defenders confirmed the malicious domain was blocked at the secure web gateway.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [173:252] "Defenders confirmed the malicious domain was blocked at the secure web gateway."
- Bias Flags: none
