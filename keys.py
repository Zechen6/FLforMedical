import utils.Encrypt as enc
from ecdsa import util, SECP256k1, VerifyingKey, ellipticcurve, SigningKey

class Public_Key():
    def __init__(self, K_x, K_y, G_x, G_y, curve) -> None:
        self.x = K_x
        self.y = K_y
        self.G_x = G_x
        self.G_y = G_y
        self.curve = curve
        self.param_list = {'curve', 'G_y', 'G_x', 'K_y', 'K_x'}
    
    @classmethod
    def clean_init(cls):
        return cls(None, None, None, None, None)
    
    def convert_into_dict(self):
        self.param_dict = {}
        for name in dir(self):
            value = getattr(self, name)
            if name in self.param_list:
                if isinstance(value, str) is False:
                    self.param_dict[name] = str(value)
                else:
                    self.param_dict[name] = value

    def convert_dict2obj(self):
        for name in dir(self):
            value = getattr(self, name)
            if name in self.param_list:
                setattr(self, name, self.param_dict[name])

class EccKeys():
    def __init__(self, pub_key:Public_Key, pri_key:SigningKey) -> None:
        self.pub_point = pub_key
        self.pri_key = pri_key


class Keys():
    def __init__(self, ecc_keys, rsa_keys) -> None:
        self.ecc_keys = ecc_keys
        self.rsa_keys = rsa_keys