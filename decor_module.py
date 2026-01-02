# decorators module


#------Decorators For EDA-------------------------------------------------------------------------------

def data_descr_decorator(f):
    def wrapper(df):
        print('View the data types and basic statistics before the pre-processing')
        res = f(df)
        print('Ready to proceed to the next stage...')
        return res
    return wrapper

def hist_decorator(f):
    def wrapper(df):
        print('View the data types and basic statistics before the pre-processing')
        res = f(df)
        print('Ready to proceed to the pre-processing...')
        return res
    return wrapper
    
def heatmap_decorator(f):
    def wrapper(df):
        print('This is a a heatmap as part of exploratory data analysis.')
        res = f(df)
        print('This is the end of the heatmap diagram...')
        return res
    return wrapper
    
def whisker_box_decorator(f):
    def wrapper(df):
        print('This is a whisker box diagram as part of exploratory data analysis..')
        res = f(df)
        print('The end of the whisker box diagram...')
        return res
    return wrapper