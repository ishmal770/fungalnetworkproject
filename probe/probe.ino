<<<<<<< HEAD:probe.ino
no
=======
#include <Arduino.h>

const int sensorPin = A0;

void setup() {
    Serial.begin(9600);
}

void loop() {
    int reading = analogRead(sensorPin);

    Serial.print("Raw: ");
    Serial.println(reading);

    delay(500);
}
   
>>>>>>> 5d40ca5cc89d51659a5ac7fb8eb0a85adf9484b4:probe/probe.ino
