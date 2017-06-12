assert __name__ == "__main__"
from roca import *
from revoice import *
from roca.common import *

dbPath = "./renri"
phoneList = [
    # (name, begin, end, votRatio)
    ("_a_a", 0.0, 1.0, 1.0),
    ("a_a_i", 1.0, 2.0, 1.0),
    ("a_i_a", 2.0, 3.0, 1.0),
    ("i_a_u", 3.0, 4.0, 1.0),
    ("a_u_e", 4.0, 5.0, 1.0),
    ("u_e_a", 5.0, 6.0, 1.0),
    ("e_a_", 6.0, 7.0, 1.0),
]

f0MatchPair = [
    np.linspace(0.0, 9.0, 4096),
    (np.hanning(4096) + 1.0) * 220.0,
]

energyMatchPair = [
    (0.0, 3.0),
    (1.0, 1.0)
]

db = voicedb.DevelopmentVoiceDB(dbPath)

print("Start synthing...")
syn = synther.Synther(db, 44100.0)
phoneList, energyMatchPair, f0MatchPair = syn.preprocess(phoneList, energyMatchPair, f0MatchPair)
syn.synth(phoneList, energyMatchPair, f0MatchPair)
