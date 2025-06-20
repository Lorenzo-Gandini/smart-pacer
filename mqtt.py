import paho.mqtt.client as mqtt
import json
from datetime import datetime
import time

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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

        phase = payload.get("phase", "unknown")
        action = payload.get("action", "").upper()
        hr_zone = payload.get("hr_zone", "?")
        power_zone = payload.get("power_zone", "?")
        fatigue = payload.get("fatigue", "unknown").upper()
        reward = payload.get("reward", "?")
        slope = payload.get("slope", "?")

        raw_seconds = int(payload.get("timestamp", 0))
        mins, secs = divmod(raw_seconds, 60)
        timestamp = f"{mins:02d}:{secs:02d}"

        return (
            f"\n━━━━━━━━━━━━━━━ 🕒 Time: {timestamp} ━━━━━━━━━━━━━━━━━━\n"
            f"{phase_icons.get(phase)} Phase   : {phase.upper():<11}  ⛰️  Slope      : {slope}\n"
            f"🎯 Action  : {action:<12} 💢 Fatigue    : {fatigue}\n"
            f"❤️  HR Zone : {hr_zone:<5}        ⚡ Power Zone : {power_zone}\n"
            f"🎁 Reward  : {float(reward):+.2f}" if reward != "?" else "🎁 Reward     : ?"
            f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(PacerLogger.format_message(payload))
        
    except Exception as e:
        print(f"❌ Error in parsing the message: {e}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("\n\n ✅ Connected to MQTT broker successfully!")
        print("🏃💨  Starting training...")
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