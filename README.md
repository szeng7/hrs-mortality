# Status

### Next Steps:

- [ ] 1) Add intermediary WD points to explore how fairness metrics degrade more granularly (seeing if it sharply drops off at a certain point (threshold) or if it deteriorates linearly)
  - [ ] a) Recreate more train sets withholding and adding different properties (geography, temporal/years, income, etc) to create more WD data points
  - [ ] b) Retrain and re-test (KEEP DATASETS IN PICKLE FOR FUTURE USE WITH DIFF CLASSIFIER/FAIRNESS METHODS)
  - [ ] c) Graph and analyze the behavior of the fairness metrics and see if any conclusions can be made.

- [ ] 2) Use different classifiers (neural networks) that can hopefully take advantage of the history of the features or do some extra feature creating to display the difference in income, health, etc over the past couple years for each individual. See if that can bump up accuracy
  - [ ] a) Use previous datasets of intermediary WDs to see how fairness changes with other classifiers

- [ ] 3) Investigate group base rates and feature interactions to clarify why statistical parity difference behaves differently than the other fairness metrics Look at relationships between fairness metrics to understand their interdependencies and underlying drivers.
  - [ ] a) Look into if specific groups are experiencing more shift than others, causing for different fairness metrics to act differently.
  - [ ] b) See if different fairness metrics follow the same trajectory or if not, why


- [ ] 4) Use fairness-enhancing techniques and assess their effectiveness under distribution shiftss
  - [ ] a) Use previous datasets of intermediary WDs to see how fairness changes

### Analysis

In a perfect world, the WD between the train and OOD test should be larger than that between the train and ID test since the OOD test is meant to represent a larger distributional shift. We see that this is only the case for some factors, such as income and race.

The fairness values should all tend to be larger/increase as the WD is larger then, indicating that the model is less fair as the distributional shift is more aggressive. 

This appears to be the case for max calibration gap because calibration depends on how well the model's predicted probabilities align with observed outcomes. Larger distribution shifts (ie higher WD) indicate that the OOD test set has feature distributions that differ significantly from those in the train set, causing for the model to become over or underconfident for certain groups, especially for those underrepresented in the training set.

This also appears to be the case for equalized odds difference because larger WDs as a result of distributional shifts can lead to imbalances in error rates (FPR and FNR) across sensitive groups. Equalized odds difference measures the difference in these error rates and larger WD often amplifies these differences as the model struggles to generalize fairly across shifted distributions.

But this doesn't appear to be the case for statistical parity difference because positive prediction rates for groups remain stable, even under large distributional shifts. Group base rates (P(y = 1 | group)) do not appear to shift proportionally with WD and 
the model may be biased or overfitted to the training data, assigning predictions uniformly across groups.

# Variables

Identifiers (these two together identify a unique individual):

- 'HHID' - Household Identifier
- 'PN' - Person Number

All properties:

- 'rmonth_survey' - month of survey
- 'ryear_survey' - year of survey
- 'age' - age of subject
- 'live_nursing_facility' - if live in nursing facility. 1 - yes, 5 - no, 8 - don't know, 9 - refused
- 'state_nursing_facility' - state that nursing facility is in. see chart below for notation for all geography related fields
- 'year_born' - year subject was born
- 'born_us' - if born in US
- 'state_born' - state if born in US (see chart)
- 'country_born' - country if not born within US, remove since no special categories
- 'education' - highest grade of education finished (see chart)
- 'highest_degree' - highest degree achieved (1 - less than bachelors, 2 - bachelors, 3 - masters/mba, 6 - law/phd/md, 7 - other, 8 - don't know, 9 - refused)
- 'race_latino_hispanic' - 1- latino/hispanic or 5 - not
- 'race' - race (1 - white, 2 - black, 7 - american indian, alaskan native, asian and pacific islander, 8 - don't know, 9 - refused)
- 'num_children' - number of children
- 'religion' - religion (1 - protestant, 2 - catholic, 3 - jewish, 4 - no pref, 7 - other, 8 - don't know, NA - not ascertained, 9 - refused)
- 'divorce_widow_status' - 1 - divorced, 2 - widowed, 5 - no, 8 - don't know, 9 - refused
- 'marital_status' - 1 - married, 3 - separated, 4 - divorced, 5 - widowed, 6 - never married, 7 - other, 8 - don't know, 9 - refused
- 'health_status' - health, see Health
- 'health_status_compared_to_prev' - health, see Health Compared
- 'health_status_better_compared_to_prev_degree' - 1 much better, 2 somewhat better
- 'health_status_worse_compared_to_prev_degree' - 4 - somewhat worse, 5 - much worse
- 'high_blood_pressure' - See Health Impairments
- 'high_blood_pressure_compared_to_prev' - See Health Comparisons
- 'diabetes' - See Health Impairments
- 'diabetes_compared_to_prev' - See Health Comparisons
- 'cancer' - See Health Impairments
- 'cancer_compared_to_prev' - See Health Comparisons
- 'lung_disease' - See Health Impairments
- 'lung_disease_compared_to_prev' - See Health Comparisons
- 'heart_condition' - See Health Impairments
'heart_condition_compared_to_prev' - See Health Comparisons
- 'stroke' - See Health Impairments
- 'heart_condition_compared_to_prev' - See Health Comparisons
- 'psychiatric_emotional_problem' - See Health Impairments
- 'psychiatric_emotional_problem_compared_to_prev' - See Health Comparisons
- 'arthritis' - See Health Impairments
- 'arthritis_compared_to_prev' - See Health Comparisons
- 'pain' - troubled with pain 1 - yes, 5 - no
- 'pain_degree' - degree of pain (1 - mild, 2 - moderate, 3 - severe)
- 'exercise' - vigorous exercise 3 times a week or more on average in past year (1 - yes, 5 - no)
- 'smoke' - 1 - yes, 5 - no
- 'still_smoke' - 1 - yes, 5 - no
- 'alcohol' - 1 - yes, 5 - no
- 'num_drinks_days_per_week'
- 'num_drinks_per_day'
- 'shortness_breath' - 1 - yes, 5 - no
- 'fatigue' - 1 - yes, 5 - no
- 'cough' - 1 - yes, 5 - no
- 'depressed_x/depressed_past_year' - depressed within previous year, 1 - yes, 5 - no
- 'tired', - 1 - yes, 5 - no
- 'appetite_loss' - 1 - yes, 5 - no
- 'appetite_increase' - 1 - yes, 5 - no
- 'memory_x/memory_self_reported' - 1 - excellent, 2 - very good, 3 - good, 4 - fair, 5 - poor,
- 'depressed_y/depressed_past_week' - depressed within previous week, 1 - yes, 5 - no
- 'children_nearby'  - 1 - yes, 5 - no
- 'dependents'  - 1 - yes, 5 - no
- 'num_dependents'
- 'type_house' - 1 - mobile home, 2 - one family home, 3 - two family home/duplex, 4 - apartment/townhouse, 7 - including coop; rooming house; recreational
                           vehicle; motor home; van; car; boat; barn; convent;
                           jail/prison; villa; in transition; garage; HUD housing;
                           trailer; motel; orphanage
- 'own_or_rent' - 1 own/buying, 2 - rent, 3 - lives rent free with relative/employer/friend
- 'property_value'
- 'year_property_acquired'
- 'vocab_level' - 1 below average, 2 average, 3 above average
- 'friendliness' - 1 hostile, 2 neutral, 3 friendly
- 'attentiveness' - 1 not attentive, 2 somehwat attentive, 3 very attentive
- 'cooperation' - 1 excellent, 2 good, 3 fair, 4 poor
- 'tiredness' - how tiring did the interview seem to be, 1 very tiring, 2 a little tiring, 3 not tiring
- 'memory_y/memory_interviewer_reported' - 1 no difficulty, 2 a little difficulty, 3 some, 4 a lot, 5 could not do at all
- 'hearing' - 1 no difficulty, 2 a little difficulty, 3 some, 4 a lot, 5 could not do at all
- 'employment_status' - 1 working now, 2 unemployed and looking, 3 temporarily laid off/sick/on leave, 4 disabled, 5 retired, 6 homemaker
- 'year_retired'
- 'state_live' - See Regions
- 'gender' - 1 male, 2 female
- 'financial_expectation' - 0 (no chance) to 100 (certain) on chances that income will keep up with inflation for the next 4 years
- 'leave_inheritance' - 0 (no chance) to 100 (certain) on leaving any inheritance
- 'income'
- 'debt_amount'
- 'wave' - slightly related to year
- 'Key' - unique identifier using HHID and PN
- 'year_death' - year of death as per label
- 'mortality_ten_years' - if patient died in the next 10 years
- 'mortality_five_years' - if patient died in the next 5 years


## Regions

1 - Northeast Region: New England Division (ME, NH, VT, MA, RI, CT)
2 - Northeast Region: Middle Atlantic Division (NY, NJ, PA)
3 - Midwest Region: East North Central Division (OH, IN, IL, MI,
    WI)
4 - Midwest Region: West North Central Division (MN, IA, MO, ND,
    SD, NE, KS)
5 - South Region: South Atlantic Division (DE, MD, DC, VA, WV, NC,
    SC, GA, FL)
6 - South Region: East South Central Division (KY, TN, AL, MS)
7 - South Region: West South Central Division (AR, LA, OK, TX)
8 - West Region: Mountain Division (MT, ID, WY, CO, NM, AZ, UT, NV)
9 - West Region: Pacific Division (WA, OR, CA, AK, HI)
10 - U.S., NA state
11 - Foreign Country: Not in a Census Division (includes
    U.S.territories)
96 - Same State (see questionnaire)
97 - OTHER COUNTRY
98 - DK (Don't Know); NA (Not Ascertained)
99 - RF (refused)
Blank - INAP (Inapplicable)

## Education

0 - No formal education
1-11 - Grade school
12 - High school
13-15 - Some college
16 - College grab
17 - Post College
97 - Other

## Health

1 - Excellent
2 - Very good
3 - Good
4 - Fair
5 - Poor
8 - Don't Know
9 - Refused

## Health Impairments

1 - Yes
3 - Disputes previous wave record but now has condition
4 - Disputes previous wave record, does not have condition
5 - No
8 - Don't Know
9 - Refused

## Health - Compared

1 - Better
2 - Same
3 - Worse
8 - Don't Know
9 - Refused