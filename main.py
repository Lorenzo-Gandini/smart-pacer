from runner_env import RunnerEnv, load_json

athletes = load_json("athletes.json")
trainings = load_json("trainings.json")
track = load_json("track_data.json")

athlete = athletes["elite"]
training = trainings["progressions"]

# Inizializza ambiente con tracciato
env = RunnerEnv(athlete, training, track_data=track, verbose=True)

state = env.reset()
done = False
total_reward = 0

actions = ['slow down', 'keep going', 'accelerate']  # logica semplificata: scegli sempre 'mantieni'

while not done:
    action = 'keep going'  # puoi cambiare questa logica con qualcosa di più smart
    state, reward, done = env.step(action)
    total_reward += reward

print(f"\n🏁 Reward totale accumulato: {total_reward:.2f}")
