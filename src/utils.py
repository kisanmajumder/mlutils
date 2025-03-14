import pandas as pd
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Dict

def outlier_value(data:pd.DataFrame, column:str, lower_q:float,upper_q:float,outlier_type:str)->Dict:
    """

    Args:
        data (pd.DataFrame): input dataframe
        column (str): column name
        lower_q (float): lower quantile vale
        upper_q (float): upper quantile value
        outlier_type (str): min. or max. value

    Returns:
        dict: outlier value and count of values crossing outlier value
    """
    result = {}
    Q1 = data[column].quantile(lower_q)
    Q3 = data[column].quantile(upper_q)
    IQR = 1.5*(Q3 - Q1)
    if outlier_type == 'min':
        result['outlier'] =  data[ (data[column] <  Q1 - IQR)][column].max()
        result['outlier_count'] = data[ (data[column] <  Q1 - IQR)][column].count()
        return result
        
    elif outlier_type == 'max':
        result['outlier'] =  data[ (data[column] >  Q3 + IQR)][column].min()
        result['outlier_count'] = data[ (data[column] >  Q3 + IQR)][column].count()
        return result

def data_profile_numeric(data:pd.DataFrame,target_col:str)->pd.DataFrame:
    """

    Args:
        data (pd.DataFrame): raw training data
        target_col(str): Name of the target column

    Returns:
        pd.DataFrame: Data profile on all the numeric columns
    """
    df = data.drop([target_col],axis=1)
    numeric_params = ['mean', 
                      'std', 
                      'min', 
                      '5%',
                      '25%',
                      '50%', 
                      '75%', 
                      '95%',
                      'max',
                      'pct_missing',
                      'skewness',
                      'kurtosis',
                      'max_outlier_info',
                      'min_outlier_info',
                      'pct_of unique_values'
                      ]
    result = {}
    for i in df.columns:
        if df[i].dtype != 'object':
            result[i] = [df[i].mean(), 
                         df[i].std(), 
                         df[i].min(), 
                         df[i].quantile(.05), 
                         df[i].quantile(.25), 
                         df[i].quantile(.5), 
                         df[i].quantile(.75), 
                         df[i].quantile(.95), 
                         df[i].max(), 
                         df[i].isnull().mean() * 100, 
                         df[i].skew(),
                         df[i].kurt(),
                         outlier_value(df,i,0.25,0.75,'max'),
                         outlier_value(df,i,0.25,0.75,'min'),
                         (df[i].nunique()*100)/df[i].count()
                        ]
    
    return pd.DataFrame(result, index=numeric_params)  

def data_profile_categorical(data:pd.DataFrame,target_col:str)->pd.DataFrame:
    """
    Args:
        data (pd.DataFrame): raw training data
        target_col (str): Name of the target col

    Returns:
        pd.DataFrame: Data profile on all the categorical columns
    """
    

    df = data.drop([target_col],axis=1)
    cat_params = ['mode',
                 'mode_freq',
                  '2nd_mode',
                  '2nd_mode_freq',
                  'pct_missing',
                  'pct_unique_values',
                  'count_unique_values']
    result = {}
    for i in df.columns:
        if df[i].dtype == 'object':
            result[i] = [df[i].mode().values[0], 
                         (df[i].value_counts().values[0]), 
                         df[i].value_counts().index[1], 
                         (df[i].value_counts().values[1]), 
                         df[i].isnull().mean() * 100,
                        (df[i].nunique()*100)/df[i].count(),
                        df[i].nunique()]
    
    return pd.DataFrame(result, index=cat_params)

def remove_duplicates(data:pd.DataFrame,col=None)->pd.DataFrame:
    """

    Args:
        data (pd.DataFrame): input dataframe
        col (str): Identify duplicates based on selected column

    Returns:
        pd.DataFrame: Duplicate data removed
    """
    if col:
        df_subset = data.drop_duplicates(subset=[col],keep='first')
    else:
        df_subset = data.drop_duplicates(keep='first')
    return df_subset
    

def remove_variables(data:pd.DataFrame,target_col:str,missing_values_threshold:float,min_variance=0):
    """
    Args:
        data (pd.DataFrame): input data 
        target_col (str): target col
        missing_values_threshold (float): missing value threshold
        
    """
    df = data.drop([target_col],axis=1)
    remove_columns = []
    for i in df.columns:
        if df[i].isnull().mean()  > missing_values_threshold:
            remove_columns.append(i)
        elif df[i].dtype == 'object' and df[i].nunique() == 1:
            remove_columns.append(i)
        elif df[i].dtype != 'object' and df[i].std() == min_variance:
            remove_columns.append(i)
            
    df_subset = data.drop(remove_columns,axis=1)
    return df_subset

class TargetSmoothedEncoding(BaseEstimator, TransformerMixin):

    def __init__(self, column_names: List[str], m: int):
        """
        Args:
            column_names (List[str]): list column names 
            m (int): smoothing hyperparameter
            target_col (str): target col name
        """
        self.column_names = column_names
        self.m = m
        self.global_mean = 0.5
        self.encoded_vals = {}
        #self.target_col = target_col

    def fit(self, X: pd.DataFrame,y: pd.Series):
        self.global_mean = y.mean()
        # Create a temporary dataframe with X and y for easier calculations
        temp_df = X.copy()
        temp_df['target'] = y
        
        for col in self.column_names:
            categorical_mean = temp_df.groupby(col)['target'].mean()
            categorical_freq = temp_df.groupby(col)[col].count()
            smoothing_factor = categorical_freq / (categorical_freq + self.m)
            self.encoded_vals[col] = smoothing_factor * categorical_mean + (1 - smoothing_factor) * self.global_mean
        return self

    def transform(self, X: pd.DataFrame):
        X_transformed = X.copy()
        for col in self.column_names:
            X_transformed[col + '_encoded'] = X_transformed[col].map(self.encoded_vals.get(col)).fillna(self.global_mean)
        return X_transformed
    
    
    
def feature_selection_wrapper(data:pd.DataFrame,type:str,target_col:str)->pd.DataFrame:
    """
    Function to do supervised feature selection using wrapper methods
    args:
    data: Cleaned data, having no missing values, outliers, and non numeric data type
    type: Type of the problem, regression or classification
    target_col: Target column name
    """
    
    features_df = data.drop(target_col,axis=1)
    cnt_features = len(features_df.columns)
    model_performance = []
    
    xtrain,xtest,ytrain,ytest = train_test_split(features_df,data[target_col],test_size=0.2,random_state=42)

    if type == 'classification':
        wrapper_model = DecisionTreeClassifier(
            criterion='entropy',
            min_samples_leaf=1,
            splitter='best'
        )
    elif type == 'regression':
        wrapper_model = DecisionTreeRegressor(
            min_samples_leaf=1,
            split='best'
        )
    wrapper_model.fit(xtrain,ytrain)
    #Calculate the F1 score of the model in the test set
    y_pred = wrapper_model.predict(xtest)
    feature_importances = pd.DataFrame(wrapper_model.feature_importances_,index=xtrain.columns,columns=['importance']).sort_values('importance',ascending=False)
    while cnt_features > 1:
        cnt_features = cnt_features-1
        features_subset = feature_importances['importance'][:cnt_features].index.to_list()
        xtrain,xtest,ytrain,ytest = train_test_split(features_df[features_subset],data[target_col],test_size=0.2,random_state=42)
        wrapper_model.fit(xtrain,ytrain)
        y_pred = wrapper_model.predict(xtest)
        if type== 'regression':
            model_performance.append(mean_squared_error(ytest,y_pred))
        elif type == 'classification':
            model_performance.append(f1_score(ytest,y_pred))
        
    if type == 'regression':
        best_model_cnt_features = model_performance.index(min(model_performance))
    elif type == 'classification':
        best_model_cnt_features = model_performance.index(max(model_performance))
        
    features_subset = feature_importances['importance'][:best_model_cnt_features].index.to_list()
    return features_subset   
        
            
