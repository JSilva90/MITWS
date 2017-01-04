import random

def randomSubsetGeneration(adam_vector, n_tests):
    ##based on paper: http://download.springer.com/static/pdf/75/art%253A10.1007%252Fs00453-006-1220-3.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Farticle%2F10.1007%2Fs00453-006-1220-3&token2=exp=1476363994~acl=%2Fstatic%2Fpdf%2F75%2Fart%25253A10.1007%25252Fs00453-006-1220-3.pdf%3ForiginUrl%3Dhttp%253A%252F%252Flink.springer.com%252Farticle%252F10.1007%252Fs00453-006-1220-3*~hmac=377a86192787a4972342f10d468e1b770734ac7767d3387a8657face5c34cc3b
    fts = []
    for i in range(0,len(adam_vector)):
        if adam_vector[i] > random.randint(1, n_tests):
            fts.append(i)
    if len(fts) == 0: ##prevent generate an empty random set
        randomSubsetGeneration(adam_vector, n_tests)
    return fts
        
        
def divide_list(n_proc, l):
    division = []
    for i in range(0, n_proc):
        division.append([])
    aux = 0
    for i in range(0, len(l)):
        division[aux].append(l[i])
        aux += 1
        aux = aux % n_proc
    return division    