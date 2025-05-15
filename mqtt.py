# mqtt.py

import paho.mqtt.client as mqtt
import json

broker = "broker.emqx.io"
topic = "smartpacer/action"

# Callback quando riceve un messaggio
def on_message(client, userdata, msg):
    try:
        print(f"📨 Raw messaggio ricevuto: {msg.payload}")
        payload = json.loads(msg.payload.decode())
        print(f"\n📥 Ricevuto:")
        print(f"⏱️ Minuto: {payload['minute']} | 🧠 Fase: {payload['phase']} | 💪 Fatica: {payload['fatigue']} | 🚀 Azione: {payload['action']}")
    except Exception as e:
        print(f"Errore nel parsing del messaggio: {e}")

# Setup client
client = mqtt.Client()
client.on_message = on_message

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("🔗 Connesso al broker con successo.")
        client.subscribe(topic)
        print(f"📡 Sottoscritto al topic: {topic}")
    else:
        print(f"❌ Connessione fallita. Codice: {rc}")


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print(f"🔌 Connessione al broker MQTT {broker}...")
client.connect(broker, 1883, 60)


print("💬 In ascolto dei messaggi... (CTRL+C per uscire)")
client.loop_forever()
