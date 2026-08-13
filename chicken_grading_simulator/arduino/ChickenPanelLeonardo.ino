#include <Keyboard.h>

// Light outputs drive MOSFET gates.
// Use one MOSFET per load:
// D5 -> Green MOSFET Gate
// D6 -> Yellow MOSFET Gate
// D7 -> Red MOSFET Gate
// D8 -> Buzzer MOSFET Gate
//
// MOSFET low-side wiring:
// 24V+ -> load +
// load - -> MOSFET Drain
// MOSFET Source -> 24V-
// Arduino GND -> 24V-
// Arduino pin -> 220 ohm -> MOSFET Gate
// MOSFET Gate -> 10k ohm -> 24V-
//
// External buttons:
// D2 -> inference button -> GND, sends keyboard "r"
// D3 -> scanner focus button -> GND, sends keyboard "n"
// D4 -> page cycle button -> GND, sends keyboard "p"

const int BUTTON_PIN = 2;
const int QR_BUTTON_PIN = 3;
const int PAGE_BUTTON_PIN = 4;
const int GREEN_PIN = 5;
const int YELLOW_PIN = 6;
const int RED_PIN = 7;
const int BUZZER_PIN = 8;

const unsigned long DEBOUNCE_MS = 50;
const unsigned long REPEAT_LOCKOUT_MS = 800;
const unsigned long PAGE_DEBOUNCE_MS = 20;
const unsigned long PAGE_REPEAT_LOCKOUT_MS = 50;

bool lastButtonReading = HIGH;
bool stableButtonState = HIGH;
unsigned long lastDebounceTime = 0;
unsigned long lastTriggerTime = 0;

bool lastQrButtonReading = HIGH;
bool stableQrButtonState = HIGH;
unsigned long lastQrDebounceTime = 0;
unsigned long lastQrTriggerTime = 0;

bool lastPageButtonReading = HIGH;
bool stablePageButtonState = HIGH;
unsigned long lastPageDebounceTime = 0;
unsigned long lastPageTriggerTime = 0;

String serialBuffer = "";
unsigned long lightOffAt = 0;

void allOutputsOff() {
  digitalWrite(GREEN_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(RED_PIN, LOW);
  digitalWrite(BUZZER_PIN, LOW);
  lightOffAt = 0;
}

void setLight(const String &color) {
  digitalWrite(GREEN_PIN, LOW);
  digitalWrite(YELLOW_PIN, LOW);
  digitalWrite(RED_PIN, LOW);
  lightOffAt = 0;

  if (color == "GREEN") {
    digitalWrite(GREEN_PIN, HIGH);
  } else if (color == "YELLOW") {
    digitalWrite(YELLOW_PIN, HIGH);
  } else if (color == "RED") {
    digitalWrite(RED_PIN, HIGH);
  } else if (color == "OFF") {
    allOutputsOff();
  }
}

void setLightFor(const String &color, unsigned long durationMs) {
  setLight(color);
  if (color != "OFF" && durationMs > 0) {
    lightOffAt = millis() + durationMs;
  }
}

void buzz(unsigned long durationMs) {
  digitalWrite(BUZZER_PIN, HIGH);
  delay(durationMs);
  digitalWrite(BUZZER_PIN, LOW);
}

void blinkAllLights(int times) {
  for (int i = 0; i < times; i++) {
    digitalWrite(GREEN_PIN, HIGH);
    digitalWrite(YELLOW_PIN, HIGH);
    digitalWrite(RED_PIN, HIGH);
    delay(250);
    digitalWrite(GREEN_PIN, LOW);
    digitalWrite(YELLOW_PIN, LOW);
    digitalWrite(RED_PIN, LOW);
    delay(250);
  }
  allOutputsOff();
}

void triggerInferenceHotkey() {
  Keyboard.press('r');
  delay(50);
  Keyboard.release('r');
  Serial.println("BUTTON:R");
}

void triggerScannerFocusHotkey() {
  Keyboard.press('n');
  delay(50);
  Keyboard.release('n');
  Serial.println("BUTTON:N");
}

void triggerPageCycleHotkey() {
  Keyboard.write('p');
  Serial.println("BUTTON:P");
}

void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command == "PING") {
    Serial.println("PONG");
  } else if (command == "LIGHT:GREEN") {
    setLight("GREEN");
    Serial.println("OK:LIGHT:GREEN");
  } else if (command == "LIGHT:YELLOW") {
    setLight("YELLOW");
    Serial.println("OK:LIGHT:YELLOW");
  } else if (command == "LIGHT:RED") {
    setLight("RED");
    Serial.println("OK:LIGHT:RED");
  } else if (command == "LIGHT:OFF") {
    allOutputsOff();
    Serial.println("OK:LIGHT:OFF");
  } else if (command.startsWith("LIGHT_FOR:")) {
    int firstColon = command.indexOf(':');
    int secondColon = command.indexOf(':', firstColon + 1);
    if (secondColon < 0) {
      Serial.println("ERR:LIGHT_FOR_FORMAT");
      return;
    }
    String color = command.substring(firstColon + 1, secondColon);
    unsigned long durationMs = command.substring(secondColon + 1).toInt();
    setLightFor(color, durationMs);
    Serial.print("OK:LIGHT_FOR:");
    Serial.print(color);
    Serial.print(":");
    Serial.println(durationMs);
  } else if (command.startsWith("BUZZ:")) {
    unsigned long durationMs = command.substring(5).toInt();
    if (durationMs == 0) {
      durationMs = 200;
    }
    buzz(durationMs);
    Serial.println("OK:BUZZ");
  } else if (command == "TEST") {
    blinkAllLights(3);
    Serial.println("OK:TEST");
  } else {
    Serial.print("ERR:UNKNOWN:");
    Serial.println(command);
  }
}

void updateTimedLight() {
  if (lightOffAt > 0 && (long)(millis() - lightOffAt) >= 0) {
    allOutputsOff();
  }
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        handleCommand(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
      if (serialBuffer.length() > 80) {
        serialBuffer = "";
        Serial.println("ERR:COMMAND_TOO_LONG");
      }
    }
  }
}

void readButton() {
  bool reading = digitalRead(BUTTON_PIN);

  if (reading != lastButtonReading) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > DEBOUNCE_MS) {
    if (reading != stableButtonState) {
      stableButtonState = reading;
      if (stableButtonState == LOW && (millis() - lastTriggerTime) > REPEAT_LOCKOUT_MS) {
        triggerInferenceHotkey();
        lastTriggerTime = millis();
      }
    }
  }

  lastButtonReading = reading;
}

void readQrButton() {
  bool reading = digitalRead(QR_BUTTON_PIN);

  if (reading != lastQrButtonReading) {
    lastQrDebounceTime = millis();
  }

  if ((millis() - lastQrDebounceTime) > DEBOUNCE_MS) {
    if (reading != stableQrButtonState) {
      stableQrButtonState = reading;
      if (stableQrButtonState == LOW && (millis() - lastQrTriggerTime) > REPEAT_LOCKOUT_MS) {
        triggerScannerFocusHotkey();
        lastQrTriggerTime = millis();
      }
    }
  }

  lastQrButtonReading = reading;
}

void readPageButton() {
  bool reading = digitalRead(PAGE_BUTTON_PIN);

  if (reading != lastPageButtonReading) {
    lastPageDebounceTime = millis();
  }

  if ((millis() - lastPageDebounceTime) > PAGE_DEBOUNCE_MS) {
    if (reading != stablePageButtonState) {
      stablePageButtonState = reading;
      if (stablePageButtonState == LOW && (millis() - lastPageTriggerTime) > PAGE_REPEAT_LOCKOUT_MS) {
        triggerPageCycleHotkey();
        lastPageTriggerTime = millis();
      }
    }
  }

  lastPageButtonReading = reading;
}

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(QR_BUTTON_PIN, INPUT_PULLUP);
  pinMode(PAGE_BUTTON_PIN, INPUT_PULLUP);
  pinMode(GREEN_PIN, OUTPUT);
  pinMode(YELLOW_PIN, OUTPUT);
  pinMode(RED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  allOutputsOff();
  Serial.begin(115200);
  Keyboard.begin();
}

void loop() {
  updateTimedLight();
  readButton();
  readQrButton();
  readPageButton();
  readSerialCommands();
}
