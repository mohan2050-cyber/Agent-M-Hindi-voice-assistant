#include<DHT.h>
#define DHTPIN 2
#define DHTTYPE DHT22

DHT dht(DHTPIN, DHTTYPE);

const int pin_pq = 7;
const int pin_rs = 8;

float readings[2];

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

  dht.begin();
  
  pinMode(pin_pq, OUTPUT);
  pinMode(pin_rs, OUTPUT);

  digitalWrite(pin_pq, HIGH);
  digitalWrite(pin_rs, HIGH);
}


void sense(){
  readings[0] = dht.readTemperature();
  readings[1] = dht.readHumidity();
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()>0){
    char cmd = Serial.read();
    switch(cmd){
      case 'p':
        digitalWrite(pin_pq, HIGH);
        break;
      case 'q':
        digitalWrite(pin_pq, LOW);
        break;
      case 'r':
        digitalWrite(pin_rs, HIGH);
        break;
      case 's':
        digitalWrite(pin_rs, LOW);
        break;
       case 't':
        sense();
        Serial.print("(");
        Serial.print(readings[0]);
        Serial.print(", ");
        Serial.print(readings[1]);
        Serial.print(")");
        Serial.print("\n");
        break;
    }
  }
}
