import paho.mqtt.client as mqtt
import json
from datetime import datetime
import time

broker = "broker.emqx.io"
topic = "smartpacer/action"

class PacerLogger:
    @staticmethod
    def format_message(payload):
        phase_icons = {
            "warmup": "🔥",
            "push": "💨",
            "recover": "🌿",
            "cooldown": "❄️",
            "unknown": "❓"
        }
        
        action_icons = {
            "slow down": "⬇️ ",
            "keep going": "🔄 ",
            "accelerate": "⬆️ "
        }
        
        phase = payload.get("phase", "unknown")
        mins, secs = divmod(int(payload.get("second", 0)), 60)
        hr_zone    = payload.get("hr_zone", "?")
        power_zone = payload.get("power_zone", "?")
        fatigue    = payload.get("fatigue", "unknown").upper()
    
        return (
            f"{phase_icons.get(phase)} {phase.upper()}\n"
            f"🕒 Second: {mins}:{secs:02d}\n"
            f"{action_icons.get(payload.get('action',''))} {payload.get('action','').upper()}\n"
            f"💓 HR Zone: {hr_zone}\n"
            f"🏋️ Power Zone: {power_zone}\n"
            f"😴 Fatigue: {fatigue}\n"
        )


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(PacerLogger.format_message(payload))
        print("-"*40)
    except Exception as e:
        print(f"❌ Error in parsing the message: {e}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("🏃‍♂️ SMART PACER CONNECTED 🏃‍♀️")
        print("🔊 I'm waiting for instructions...")
        print("="*50)
        client.subscribe(topic)
    else:
        print(f"❌ Connection failed. Codice: {rc}")

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, 1883, 60)
client.loop_forever()