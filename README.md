# Smart Pacing Assistant for Runner

Simulation and optimization of the pace for runners, based on MDP and Q-Learning. 


## ✅ Punto 1 – Obiettivo e contesto

- Runner singolo, su percorso reale (profilo da GPX).
- Allenamenti strutturati e noti a priori (fartlek, lungo, recupero, ecc.).
- Obiettivo: allenarsi correttamente, seguendo il piano e adattandosi alle condizioni fisiologiche (HR, potenza, fatica).
- Il sistema fornisce suggerimenti in tempo reale: `"rallenta"`, `"mantieni"`, `"accelera"`.

## ✅ Punto 2 – Stato

Lo stato del sistema è una tupla:

state = (
    HR_zone,         # zona cardiaca: Z1–Z5
    power_zone,      # zona di potenza: Z1–Z5
    fatigue_level,   # low, medium, high (score 0–10)
    segment_index,   # fase dell’allenamento
    phase_label,     # 'warmup', 'push', 'recover', 'cooldown'
    slope_level      # 'flat', 'uphill', 'steep_uphill', ecc.
)

✅ Punto 3 – Action Space
Azioni disponibili:

actions = ['rallenta', 'mantieni', 'accelera']

✅ Punto 4 – Reward Function
Reward definito come:

reward = f(state, action)

Dipende da:
- allineamento tra zona attuale e zona target
- livello di fatica
- coerenza tra fase e azione

✅ Punto 5 – Transizioni
Transizioni deterministiche con rumore controllato:

HR_zone	        cambia con azione + slope
power_zone	    cambia direttamente con azione
fatigue_level	cresce con HR, azione, slope, rumore
segment_index	avanza ogni minuto
phase_label	    derivata da segment_index
slope_level	    estratta dal GPX

Il modello di fatica è flessibile e può includere effetti di recupero, salita, tempo in zona, e adattamento all’allenamento.

