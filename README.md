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
    
Data pre-processing what attributes we use and which not:
Variables to add:
-	Activity: several studies have shown that activity has influence on mood (Peluso, M. A. M., & Andrade, L. H. S. G. D. (2005), and other studies…
-	Screentime: studies that investigated screentime influence on mood  (Sarris et al.)
-	Valence: is basically another measurement for positive and negative feelings and emotions therefore it will influence the mood
-	Arousal: okay also put it in
-	Office: work stress has an active effect on the mood -> so maybe high usage of office apps -> stress -> bad mood (Davide Carneiro, Paulo Novais, Juan Carlos         Augusto, and Nicola Payne, 2015)
-	Game: studies that casual videos game can improve the mood (Carmen V. Russoniello1, Kevin O’Brien1 and Jennifer M. Parks1, 2009)
-	Call: add (no literature but its obvious how a SMS can affect the mood of the day)
-	SMS: add (no literature but its obvious how a SMS can affect the mood of the day)
-	Social: studies that show that social media has effect on mood. (Berry N, Emsley R, Lobban F, Bucci S., 2018)
-	Entertainment: affects mood. Period.
-	Communication: 
-	Weather: see study (but yeah I know its just the weather app usage but if someone uses it then it can have an effect on the mood because they see if weather is     bad/good and then…)



Drop: unknown, utilities, other (because we don’t know what these apps really are and so don’t know how we can interpret the effect of these apps on the mood); finance: to much data missing (or nobody uses them) (discuss!!!)
