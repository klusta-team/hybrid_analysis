import hashlib as hl
import joblib_utils as ju
import string

@ju.func_cache
def hash_dictionary_md5(dict1):
    hashdict = hl.md5(repr(sorted(dict1.items(), key= lambda (key,val): key))).hexdigest()
    print hashdict
    return hashdict

def get_product_hashlist(dictlist):
    '''gives the product hash of a list of dictionaries,
       e.g. dictlist = [dict1, dict2, dict3]
       product_hash = [Hash(dict1),Hash(dict2),Hash(dict3)]
    '''
    dictlisthash = []
    for dictio in dictlist:
        dictlisthash.append(hash_dictionary_md5(dictio))
    return dictlisthash

def make_concatenated_filename(dictlisthash):
    ''' concatenate [Hash(dict1),Hash(dict2),Hash(dict3)] to give
        producthash = Hash(dict1)_Hash(dict2)_Hash(dict3)'''
    producthash = string.join(dictlisthash,'_')
    return producthash
