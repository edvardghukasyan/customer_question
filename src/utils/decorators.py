def singleton(cls):
    """
    A singleton decorator for classes.
    """
    instances = {}
    
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return getinstance
