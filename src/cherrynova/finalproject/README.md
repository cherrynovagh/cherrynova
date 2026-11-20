## 1. About the Dataset

This dataset contains **818 police-reported traffic crashes in Alabama involving a rideshare vehicle** (Uber, Lyft or a similar app-based ride service), from January 1, 2017 to June 30, 2026. Each row is one crash, described by 223 variables covering when and where the crash happened, the roadway and environmental conditions, the drivers and vehicles involved, and the crash outcome.

Alabama crash reports do not have a field that marks a vehicle as a rideshare vehicle. The crashes were therefore identified by searching the officers' written crash narratives for rideshare-related terms (for example, "uber", "lyft", "rideshare", "uber driver", "picking up a passenger","dropping off a passenger").

## 2. Original Source

The data were not collected experimentally by me. The original source is:

- **Alabama Department of Transportation (ALDOT)** crash records, based on the Alabama Uniform Traffic Crash Report completed by police officers.
- Accessed through the **CARE (Critical Analysis Reporting Environment)** portal, developed and maintained by the **Center for Advanced Public Safety (CAPS), The University of Alabama**: https://care.aladata.com

## 3. How to Obtain the Dataset
The data files can be shared privately upon request from ALDOT at crashdata@dot.state.al.us.

The dataset was obtained from CARE with the following steps:

1. Dataset: **Alabama Crash Data**
2. Date range: **01/01/2017 to 06/30/2026**
3. Narrative Search using rideshare-related terms, including: uber, lyft, rideshare, ride share, ride-share, ridesharing, ride hailing, tnc, transportation network company, uber driver, lyft driver, driving for uber, driving for lyft, working for uber, working for lyft, ride app, on the app, picking up a fare, picking up a passenger, dropping off a passenger
4. No additional filters
5. The data was then exported to an excel format with all the crash 223 crash variables.

## 4. Problem to Solve

Predicting Injury Severity of Rideshare Crashes in Alabama: The Role of Fatigue-Related Factors

Rideshare drivers often drive after working a primary job and during late-night hours. Both extended wakefulness and night driving increase the risk of drowsy driving, and the American Academy of Sleep Medicine considers fatigue and sleepiness an inherent safety risk in the ridesharing industry (Berneking et al., 2018). However, because rideshare involvement is not recorded in crash reports, little is known about how severe rideshare crashes are in Alabama or what conditions make them worse.

The goal of this project is to predict the injury severity of rideshare-involved crashes and to identify which factors are linked to more severe outcomes, with a focus on fatigue-related conditions such as late-night hours, darkness and drivers reported as asleep or fatigued.

## Data variables

The full export has 223 variables; the ones selected for this project are listed below.

| Group | Variable | Type | Example values |
|---|---|---|---|
| Fatigue-related | Time of Day | Categorical | 24 one-hour periods (grouped into night 11 PM–7 AM vs. day) |
| Fatigue-related | Lighting Conditions | Categorical | Daylight, Dawn, Dusk, Dark – Roadway Not Lighted, Dark – Spot Illumination, Dark – Continuous Lighting |
| Fatigue-related | CU Driver Condition | Categorical | Apparently Normal, Asleep/Fainted/Fatigued, Under the Influence of Alcohol/Drugs, Illness, Unknown |
| Fatigue-related | Distracted Driving Opinion | Categorical | Not Distracted, Fatigued/Asleep, Electronic Communication Device, Passenger |
| Crash | Manner of Crash | Categorical | Rear End, Single Vehicle, Sideswipe – Same Direction, Side Impact |
| Crash | Number of Vehicles | Numeric | 1, 2, 3, 4 |
| Crash | First Harmful Event | Categorical | Collision with Vehicle in Traffic, Collision with Parked Motor Vehicle, Ran Off Road, Collision with Pedestrian |
| Location | Rural or Urban | Categorical | Rural, Urban |
| Location | Highway Classifications | Categorical | Municipal, State, County, Interstate, Private Property |
| Roadway | CU Speed Limit | Numeric | Posted speed limit (25, 35, 45, 55 mph, …) |
| Roadway | Intersection Related | Categorical | Yes, No |
| Environment | Weather | Categorical | Clear, Cloudy, Rain, Mist, Fog |
| Environment | CU Roadway Condition | Categorical | Dry, Wet, Ice |
| Driver | CU Driver Raw Age | Numeric | Age in years |
| Driver | CU Driver Gender | Categorical | Male, Female |
| Driver | CU Driver Officer Opinion Alcohol / Drugs | Categorical | Yes, No |
| Time | Day of the Week, Year | Categorical | Monday–Sunday; 2017–2026 |

"CU"-causal unit.

## Label

The label is Crash Severity , recorded on the KABCO injury scale:

| Crash Severity | Count |
|---|---|
| Property Damage Only (O) | 589 |
| Possible Injury (C) | 79 |
| Non-Incapacitating Injury (B) | 84 |
| Suspected Serious Injury (A) | 31 |
| Fatal Injury (K) | 14 |
| Unknown | 21 |

Crashes with unknown severity (21) will be excluded.

## 5. Why This Is an Engineering Problem

Transportation safety engineering relies on crash data to identify high-risk conditions and to choose effective safety countermeasures, such as roadway lighting, signal timing, roadway design improvements and enforcement. State agencies use crash severity analysis to decide where safety investments will prevent the most deaths and serious injuries.

Rideshare is a growing travel mode, but its crashes are not tracked in standard crash records. Understanding which conditions make rideshare crashes more severe can support engineering and policy decisions, including lighting improvements on high-risk roads, the design of safe pickup and drop-off zones, and limits on driver hours to reduce fatigue.

## 8. Limitations

- Rideshare crashes can only be identified when the officer mentioned rideshare involvement in the narrative, so the dataset likely undercounts rideshare crashes.

## 9. References

- Alabama Department of Transportation and Center for Advanced Public Safety. *CARE User Guide 2024*. https://care.aladata.com
- Berneking, M., Rosen, I. M., Kirsch, D. B., et al. (2018). The risk of fatigue and sleepiness in the ridesharing industry: An American Academy of Sleep Medicine position statement. *Journal of Clinical Sleep Medicine*, 14(4), 683–685. https://doi.org/10.5664/jcsm.7072