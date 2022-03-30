

label2SOS = {
    'airway': 1540,
    'bile': 1540,
    'bone': 2425,
    'cartilage': 1665,
    'cns': 1540,
    'connective tissue': 1613,
    'fat': 1478,
    'gland': 1540,
    'gut': 1534,
    'heart': 1540,
    'kidney': 1572,
    'liver': 1540,
    'lung': 1540,
    'membrane': 1540,
    'muscle': 1547,
    'skin': 1613,
    'spleen': 1540,
    'vessel': 1501,
    'zero': 1540
}

label2Density = {
    'airway': 993,
    'bile': 993,
    'bone': 2062,
    'cartilage': 1100,
    'cns': 993,
    'connective tissue': 1120,
    'fat': 950,
    'gland': 993,
    'gut': 1040,
    'heart': 993,
    'kidney': 1050,
    'liver': 993,
    'lung': 993,
    'membrane': 993,
    'muscle': 1050,
    'skin': 1120,
    'spleen': 993,
    'vessel': 1040,
    'zero': 993
}


SOS_RANGE = [1478, 2425]
DENSITY_RANGE = [
    min(list(label2Density.values())),
    max(list(label2Density.values()))]