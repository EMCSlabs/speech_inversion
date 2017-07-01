def zscore(data, axis=1):
    '''
    인풋-아웃풋 데이터를 z-score normalize하기 위한 함수
    '''
    # data shape: (dimension)x(time)
    mu = np.mean(data, axis, keepdims=True)
    std = np.std(data, axis, keepdims=True)
    out = (data - mu) / std
    return out, mu, std 

def zscore_reverse(data, mu, std):
    '''
    z-score로부터 원래 데이터로 복구하는 함수
    '''
    return data*std + mu

def minmax(data, minmax_range=(0.1, 0.9), axis=1):
    '''
    인풋-아웃풋 데이터를 min, max값으로 normalize하기 위한 함수
    '''
    # data shape: (dimension)x(time)
    min_val = np.min(data, axis=1, keepdims=True)
    max_val = np.max(data, axis=1, keepdims=True)
    min_new, max_new = minmax_range
    out = (data - min_val)/(max_val - min_val)*(max_new - min_new) + min_new
    return out, (min_val, max_val, min_new, max_new)

def minmax_reverse(data, param):
    '''
    min-max normalize된 값을 복구하는 함수
    '''
    min_val, max_val, min_new, max_new = param
    range_old = max_val - min_val
    range_new = max_new - min_new
    out = (data - min_new)*range_old/range_new + min_val
    return out

def sigmoid(data):
    return 1 / (1 + np.exp(-data))

def sigmoid_reverse(data):
    return np.log(data)