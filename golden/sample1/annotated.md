# Annotated Claims

## C001
- Text: The incident response report states that credentials were reset for 12 affected user accounts.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [0:94] "The incident response report states that credentials were reset for 12 affected user accounts."
- Bias Flags: none

## C002
- Text: The team claims this catastrophic outage proves the vendor platform is always unreliable.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [95:184] "The team claims this catastrophic outage proves the vendor platform is always unreliable."
- Bias Flags:
  - alarmist: [116:128] catastrophic (fix: Use neutral phrasing for alarmist language.)
  - certainty: [166:172] always, [136:142] proves (fix: Use neutral phrasing for certainty language.)

## C003
- Text: Analysts observed DNS traffic to update-security.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [185:234] "Analysts observed DNS traffic to update-security."
- Bias Flags: none

## C004
- Text: example from two hosts.
- Score: ScoreLabel.SUPPORTED
- Evidence:
  - [234:257] "example from two hosts."
- Bias Flags: none
