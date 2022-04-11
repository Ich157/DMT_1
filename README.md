# DMT_1
Assumptions preprocessing:
First days not much values: skipped them (also first day WITH values has no mood) -> still 130 entries with no mood
If call / sms is NAN -> put as 0 for the day
Variables such as "utilities", "game", "unknown", "finance", "office", "weather", "travel" very little values -> fill NAN as 0









Temporal model: LSTM
Proposed (basic) architecture:
    Embedding layer not needed? -> numerical values already
    LSTM layer
    Linear layer