from pylinac import CatPhan504
from catpy504.classcbct import CBCT

def read_file(path: str) -> dict:
    mycbct = CatPhan504.from_zip(path)
    mycbct.analyze()
    CatPhan = CBCT(mycbct)
    #CatPhan.test()
    result = CatPhan.result()
    return [result, CatPhan, mycbct]
