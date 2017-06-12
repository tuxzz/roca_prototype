assert __name__ == "__main__"
from roca import *
from revoice import *
from roca.common import *

dbPath = "./renri"
data = [
    # (objName, name, pre, post, left, right, vot, attack, release, overlap)
    ("aaiauea", "_a_a", (), (), 0.381, 0.995, 0.381, 0.400, 0.900, 0.0),
    ("aaiauea", "a_a_i", (), (), 0.995, 1.534, 0.995, 1.000, 1.500, 0.5),
    ("aaiauea", "a_i_a", (), (), 1.534, 2.045, 1.534, 1.600, 2.000, 0.5),
    ("aaiauea", "i_a_u", (), (), 2.045, 2.486, 2.045, 2.100, 2.400, 0.5),
    ("aaiauea", "a_u_e", (), (), 2.486, 2.983, 2.486, 2.500, 2.900, 0.5),
    ("aaiauea", "u_e_a", (), (), 2.983, 3.502, 2.983, 3.000, 3.500, 0.5),
    ("aaiauea", "e_a_", (), (), 3.502, 4.379, 3.502, 3.600, 4.300, 0.5),
]

voiceDB = voicedb.DevelopmentVoiceDB(dbPath)
if(voiceDB.hasObject("config.phone")):
    phoneConf = voiceDB.loadObject("config.phone")
else:
    phoneConf = objects.PhoneConfigurationObject()

for objName, name, pre, post, left, right, vot, attack, release, overlap in data:
    assert(left >= overlap)
    f0ObjectPath, hnmObjectPath, sinEnvObjectPath = "%s.f0Obj" % (objName,), "%s.hnmObj" % (objName,), "%s.sinEnvObj" % (objName,)
    f0Object = voiceDB.loadObject(f0ObjectPath)
    meanF0 = np.mean(f0Object.f0List[f0Object.f0List > 0.0])
    phoneConf.insert(name, meanF0, pre, post, left, right, vot, attack, release, overlap, f0ObjectPath, hnmObjectPath, sinEnvObjectPath)

voiceDB.saveObject("config.phone", phoneConf)
