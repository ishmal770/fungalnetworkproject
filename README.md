# fungalnetworkproject
TO DEMO ECOSCOUT PLEASE USE THIS LINK 
https://fungalnetworkproject2git-r8umw52abah4r7vyfyhjew.streamlit.app/

### Python packages
```bash
pip install streamlit numpy pandas matplotlib pillow pyserial torch torchvision scikit-learn requests
```
 
### Optional: satellite panel
```bash
pip install earthengine-api
```
Also requires a **Google Earth Engine account** and a registered **GCP project ID**
(entered in the app's "Earth Engine project ID" field). Without both, this panel
disables itself gracefully and the rest of the app still works.
 
### Hardware / non-Python
**VS Code + PlatformIO extension** — used to build/upload `probe.ino` and monitor
  its serial output (not the standalone Arduino IDE)
**Arduino board + moisture probe**, wired to analog pin `A0`
**USB cable** with data lines (not a charge-only cable)
No external Arduino libraries needed — the sketch only uses the built-in `Arduino.h`/`Serial`
## Setup — run in this order
 
1. **Install Python dependencies** (see above).
2. **Flash the Arduino**: open `probe/probe.ino` in VS Code (PlatformIO), Save,
   then run the **Upload** task. Confirm it prints "SUCCESS" in the terminal.
3. **Calibrate the soil probe**:
```bash
   python log_moisture.py --calibrate
```
   This creates `calibration.json`, which the app requires before any soil
   sampling will work.
4. *(Optional)* **Train the CNN**:
```bash
   python fungal_model.py
```
5. **Run the app**:
```bash
   streamlit run app.py

***Only one program can hold the Arduino's serial port open at a time — Serial Monitor, PlatformIO's Monitor task, log_moisture.py, and the Streamlit app all compete for the same port.***

AI + Satellite Imagery + Hardware for Mapping Potential Fungal Activity

Fungal Network Mapper is a multi-modal environmental monitoring system that identifies areas with a higher likelihood of fungal activity by combining computer vision, satellite imagery, and physical soil measurements.

Our goal is to create a scalable way to identify and investigate potential fungal hotspots without requiring researchers to manually survey every square meter of land.

Important: Our system estimates the likelihood of fungal activity. It does not directly prove that an underground fungal network exists. Physical sampling and biological testing would be needed for definitive confirmation.

The Problem

Fungal networks, including mycorrhizal networks, play an important role in soil ecosystems, plant health, and nutrient cycling. When people build new projects on land, they can be unaware of their disruption to fungal networks. This means they must go through a lengthy and costly process when getting their permit in order to build. Our app would allow potential buyers to first survey the land and see if theres a high probability of disrupting networks. This could inform their buying decisions and make it so they don't go through difficult processes in the future.

However, underground fungal networks are difficult to map because they are:

Invisible from the surface
Distributed across large areas
Difficult and expensive to sample manually
Influenced by environmental conditions such as moisture and vegetation


Our Solution

Fungal Network Mapper combines three layers of evidence:

1. CNN model- Above-Ground Fungal Activity Detection

We train a CNN (Convolutional Neural Network) to analyze plant and environmental imagery for visible indicators associated with potential fungal activity.

Our model combines three datasets:

PlantDoc

Provides images of plants with different diseases and healthy conditions.

Plant disease can sometimes be associated with fungal pathogens, allowing the model to learn visual patterns associated with diseased vegetation.

PlantVillage

Provides a large collection of healthy and diseased plant images.

This helps the model distinguish between normal plant appearance and abnormal vegetation patterns.

iNaturalist

Provides observations of organisms in natural environments, including fungi.

This gives the model additional information about visible fungal organisms and fungal-related environmental features.

The CNN produces a fungal activity likelihood score based on observable above-ground characteristics.

2. Satellite imagery and historical data

We use historical satellite imagery and environmental data to identify areas with environmental conditions that are more likely to be associated with fungal activity.

The system analyzes spatial patterns over time and generates a heat map.

 Red = Higher likelihood
 Yellow = Moderate likelihood
 Green = Lower likelihood

The heat map allows potential buyers to quickly identify areas worth investigating. 


3. Hardware — Soil Moisture / Conductivity Verification

Once the satellite model identifies a potential hotspot, users can physically test the area using our Arduino-based sensor.

Our hardware component uses a soil sensor module connected to an Arduino to measure electrical properties of the soil that vary with moisture.

The sensor provides a relative measurement ranging from:

LOW MOISTURE  ←────────────→  HIGH MOISTURE

Soil moisture is useful because moisture availability is an important environmental factor affecting fungal growth and microbial activity. It can be prepared with the historical data and CNN model, and if all three match then theres a high likelihood of fungal activity.

 

Our three evidence sources:
Layer	What we measure	Purpose
CNN	Above-ground visual indicators	Detect potential fungal activity
Satellite	Historical environmental/spatial patterns	Find potential fungal hotspots
Arduino	Soil moisture/electrical response	Ground-check environmental conditions

When multiple signals agree, the location becomes a higher-priority candidate for biological investigation.

How It Works
Step 1 — Analyze historical data

Satellite imagery and environmental information are processed to identify spatial patterns associated with potential fungal activity.

Step 2 — Generate the heat map

Our system assigns each location a likelihood score and visualizes it geographically.

🟢 🟢 🟡 🟡 🔴 🔴
🟢 🟡 🟡 🔴 🔴 🔴
🟢 🟢 🟡 🔴 🔴 🟡
🟢 🟢 🟢 🟡 🟡 🟢
Step 3 — Select a hotspot

A user identifies a red/high-likelihood area on the map.

Step 4 — Collect physical measurements

The user visits the location and takes soil measurements using the Arduino sensor.

Step 5 — Compare the signals

The hardware measurement is uploaded to the system and associated with the corresponding geographic location.

Step 6 — Combine the evidence

The system combines:

CNN prediction + satellite prediction + physical soil measurement

to produce a more informed estimate of potential fungal activity.


📊 Example Output

A researcher could select a location on the map and see:

LOCATION: Site #047

Satellite likelihood:     82%
CNN likelihood:            74%
Soil moisture signal:     HIGH

--------------------------------
Combined priority:        HIGH
--------------------------------

Technology Stack
Machine Learning
Python
Convolutional Neural Networks (CNN)
Computer Vision
PlantDoc
PlantVillage
iNaturalist
Geospatial
Satellite imagery
Historical environmental data
Geographic heat maps
Spatial analysis
Hardware
Arduino
Soil moisture/conductivity sensor
Analog sensor readings
Ground-based environmental measurements
Application
Interactive map
Heat-map visualization
Sensor data integration
Location-based fungal activity scoring
What Makes This Different?

Most approaches to studying fungal networks rely heavily on manual field sampling.

Our approach instead uses a:

Predict → Prioritize → Ground-check → Investigate

workflow.

Rather than physically testing an enormous area, our system helps researchers determine where testing may be most valuable.

This could make large-scale ecological surveys more efficient and help researchers monitor ecosystems over time.

Future Applications

Fungal Network Mapper could eventually be used for:

Forest ecosystem monitoring
Agricultural soil health
Environmental impact assessments
Urban development planning
Ecological research
Long-term ecosystem monitoring
Targeted biological field sampling
Future Improvements

Our current hardware measures environmental conditions that may correlate with fungal activity. Future versions could add more direct biological measurements.

Potential additions include:

Soil DNA/eDNA analysis
Laboratory fungal identification
More environmental sensors
Temperature measurements
Soil pH
Electrical conductivity
Soil nutrient measurements
Higher-resolution satellite imagery
More geographically diverse training data

These additions could help us move from fungal activity prediction toward stronger biological validation.

Limitations

Our system does not currently claim to directly detect underground fungal networks.

Instead, it identifies correlated indicators and environmental conditions that can help locate areas where fungal activity may be more likely.

The strongest confirmation would require biological sampling, such as laboratory analysis of soil or root-associated fungal DNA.

This distinction is important because:

Correlation ≠ confirmation.

Our goal is to make fungal research more targeted and scalable, not to replace biological testing.



