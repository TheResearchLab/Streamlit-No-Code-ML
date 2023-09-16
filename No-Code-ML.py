import streamlit as st
from sklearn.model_selection import train_test_split
from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
import sklearn.metrics as metrics
import xgboost
import time as t
import numpy as np
import pandas as pd



def main() -> None:

    
    # Sklearn packages/module metadata

    model_selection_dict = { "Classification":{
            "LogisticRegression":"sklearn.linear_model",
            "XGBClassifier":"xgboost.sklearn",
            "SVC":"sklearn.svm",
            "KNeighborsClassifier":"sklearn.neighbors",
            "GaussianNB":"sklearn.naive_bayes",
            "BernoulliNB":"sklearn.naive_bayes",
            "LinearSVC":"sklearn.svm",
            "MultinomialNB":"sklearn.naive_bayes",
            "DecisionTreeClassifier":"sklearn.tree",
            "RandomForestClassifier":"sklearn.ensemble"
            },
            "Regression":{
            "LinearRegression":"sklearn.linear_model",
            "XGBRegressor":"xgboost.sklearn",
            "Lasso":"sklearn.linear_model",
            "ElasticNet":"sklearn.linear_model",
            "BayesianRidge":"sklearn.linear_model",
            "Ridge":"sklearn.linear_model",
            "KNeighborsRegressor":"sklearn.neighbors",
            "SVR":"sklearn.svm",
            "DecisionTreeRegressor":"sklearn.tree",
            "RandomForestRegressor":"sklearn.ensemble"
            }
    } 

    sklearn_data_dict = {"module":"sklearn.datasets",
                     "data":{
                         "Iris (Multi-Class Classification)":"load_iris",
                         "Diabetes (Regression)":"load_diabetes",
                         "Wine (Mult-Class Classification)":"load_wine",
                         "Breast Cancer (Binary Classification)":"load_breast_cancer"
                     } 
                   }

     # Data Transformations Dict
    data_preprocess_dict = {
        "StandardScaler": {"class":"StandardScaler","module_name":"sklearn.preprocessing"},
        "MinMaxScaler": {"class":"MinMaxScaler","module_name":"sklearn.preprocessing"},
    }

    ctx = get_script_run_ctx()

    #st.write(ctx.session_id)

   
    # if hasattr(st.session_state,'user_session_data'):
    #     user_session_data = st.session_state.user_session_data[ctx.session_id]
    
    if not hasattr(st.session_state,'user_session_data'):
        st.session_state.user_session_data = {}

    if ctx.session_id not in st.session_state.user_session_data:
        st.session_state.user_session_data[ctx.session_id] = {}
        
    
    
   #================== General Use Functions =====================#


    #import module class
    def import_class(module_name,class_name):
        try:
            module = __import__(module_name, globals(),locals(),[class_name])
        except ImportError:
            return None
        return vars(module)[class_name]

    
    def instantiate_obj(class_name,module_name):
        object_instance = import_class(str(module_name),str(class_name))
        return object_instance()
    
    def update_data():
        get_train_test(str(data_key),data_source)
        st.session_state.user_session_data[ctx.session_id]['data_source'] = data_source 
        st.session_state.user_session_data[ctx.session_id]['data_key'] = data_key



    def check_cache_data():

        # if there is no data then load data
        if 'data' not in st.session_state.user_session_data[ctx.session_id]:
            update_data()

        # if the dataset changed then change the dataset
        if 'data_key' in st.session_state.user_session_data[ctx.session_id] and st.session_state.user_session_data[ctx.session_id]['data_key'] != data_key:
            update_data()

        if 'data_source' in st.session_state.user_session_data[ctx.session_id] and st.session_state.user_session_data[ctx.session_id]['data_source'] != data_source:
            update_data()
        
            
        
        
    def user_data_check():
        if data_source == 'My Computer':
            st.error('Please upload csv file with your data')

    

    
    def check_cache_hyperparams():
        module_name = model_selection_dict[prediction_task][algorithm_name]

        # Check for change in algo
        if 'pred_algorithm_name' in st.session_state.user_session_data[ctx.session_id] and 'param_algorithm_name' in st.session_state.user_session_data[ctx.session_id]:
            check_1 = st.session_state.user_session_data[ctx.session_id]['pred_algorithm_name'] == st.session_state.user_session_data[ctx.session_id]['param_algorithm_name']
            same_algo_bool = st.session_state.user_session_data[ctx.session_id]['algorithm_name'] == algorithm_name and check_1
        
        elif 'param_algorithm_name' in st.session_state.user_session_data[ctx.session_id]:
            same_algo_bool = st.session_state.user_session_data[ctx.session_id]['param_algorithm_name'] == algorithm_name

        elif 'pred_algorithm_name' in st.session_state.user_session_data[ctx.session_id]:
            same_algo_bool = st.session_state.user_session_data[ctx.session_id]['pred_algorithm_name'] == algorithm_name
        
        elif 'pred_algorithm_name' not in st.session_state.user_session_data[ctx.session_id] and 'param_algorithm_name' not in st.session_state.user_session_data[ctx.session_id]:
            same_algo_bool = False

        elif st.session_state.user_session_data[ctx.session_id]['algorithm_name'] == algorithm_name:
            same_algo_bool = True
            
        # check for existing hyperparams
        if 'hyperparams' in st.session_state.user_session_data[ctx.session_id] and same_algo_bool :
            #hyperparams = user_session_data.hyperparams
            model =  model_instance(str(algorithm_name) ,str(module_name)) #can I call set params on this?
            model_param_dict = st.session_state.user_session_data[ctx.session_id]['hyperparams'] 
            if 'dual' in model_param_dict:
                model_param_dict['dual'] = False
            model = model.set_params(**model_param_dict)
            return model, model_param_dict
        else:
            model = model_instance(str(algorithm_name) ,str(module_name))
            model_param_dict = model.get_params()
            st.session_state.user_session_data[ctx.session_id]['hyperparams'] =  model_param_dict
            return model, model_param_dict
            
        


   #================== Model Instance Functions =====================#

    
    # instantiate new ml model 
    @st.cache_resource  
    def model_instance(algorithm_name,module_name):

        model = import_class(str(module_name),str(algorithm_name))
        return model()
            

    def train_model(data_key,data_source,model):
        
        if 'data' in st.session_state.user_session_data[ctx.session_id]:    
            X_train, X_test, y_train, y_test = st.session_state.user_session_data[ctx.session_id]['data']
            model.fit(X_train,y_train)
            return [model,X_test,y_test]
            
        else:
            X_train, X_test, y_train, y_test = get_train_test(str(data_key),data_source) #change function name
            model.fit(X_train,y_train)
            return [model,X_test,y_test]


   
   #================== Data Related Functions =====================#

    
    
    def get_train_test(tableName,data_source) -> list:
        
        
        def split_data(dataframe) -> None:
            X = dataframe.iloc[:,:-1]
            y = dataframe.iloc[:,-1:]
            st.session_state.user_session_data[ctx.session_id]['data'] = train_test_split(X, y, test_size=0.3) 
            st.session_state.user_session_data[ctx.session_id]['data_cols'] = dataframe.columns
            return None


        if data_source == 'My Computer':
            df = pd.read_csv(data_key)
            split_data(df)
            st.session_state.user_session_data[ctx.session_id]['feature_names'] = df.columns
            return st.session_state.user_session_data[ctx.session_id]['data']
        
        if data_source == 'Sklearn Dataset':
            #Instantiate sklearn data object
            dataset_name = sklearn_data_dict['data'][data_key]
            data_module = sklearn_data_dict['module']
            data_instance = instantiate_obj(dataset_name,data_module)
            df = pd.DataFrame(np.column_stack((data_instance['data'],data_instance['target'])),
                            columns=[*data_instance['feature_names'],'target'])
            split_data(df)
            return st.session_state.user_session_data[ctx.session_id]['data']
        
        
    
   #================== Frontend Integration =====================#
    
    
    st.sidebar.success('The Research Lab™')

    st.header(':green[No]-Code-ML')

    # Display the input values
    prediction_task = st.selectbox("Prediction Task",["Classification","Regression"])
    data_source = st.selectbox("Data Location",['My Computer','Sklearn Dataset'])
    if data_source == 'Sklearn Dataset':
        data_key = st.selectbox("Choose a Dataset",[i for i in sklearn_data_dict['data'].keys()])
    if data_source == 'My Computer':
        data_key = st.file_uploader('Upload Data as CSV')
    algorithm_name = st.selectbox("Algorithm Type", [i for i in model_selection_dict[prediction_task].keys()]) 
    

    # Create Buttons for Setting Model Parameters, Model Training, and Data Transformations 
    params_btn,transform_data_btn,train_btn,predict_btn = st.columns([0.07,0.06,0.04,0.04],gap="small")
    with params_btn:
        params_bool = st.selectbox('Set Model Params',['No','Yes'],index=0)
    with transform_data_btn:
        transform_data_bool = st.selectbox('Transform Data',['No','Yes'],index=0)
    with train_btn:
        train_model_bool = st.selectbox('Train',['No','Yes'],index=0)
    with predict_btn:
        predict_bool = st.selectbox('Predict',['No','Yes'],index=0)

    
        
    

    #Dictionary to hold new parameters
    model_param_dict = {}
    params_form = st.empty() 

    # Set Model Params Functionality
    if params_bool == 'Yes' and 'Yes' not in [transform_data_bool,train_model_bool,predict_bool]:

        with params_form.form("hyperparam_form"):
            
            _,model_param_dict = check_cache_hyperparams()
            
            
        
            st.write(f" :green[{algorithm_name}] Hyperparameters")
            
            for key,value in model_param_dict.items():
                if key == 'dual':
                    st.success(key + " must be False")
                    value = False
                original_type = 'NoneType' if isinstance(model_param_dict[key],type(None)) else type(model_param_dict[key])
                model_param_dict[key] = None if original_type == 'NoneType' else original_type(st.text_input(f"{key}",model_param_dict.get(key,value)))
            submitted = st.form_submit_button("Update Hyperparameters")

            if submitted:
                st.session_state.user_session_data[ctx.session_id]['hyperparams'] = model_param_dict
                st.session_state.user_session_data[ctx.session_id]['algorithm_name'] = algorithm_name
                st.session_state.user_session_data[ctx.session_id]['param_algorithm_name'] = algorithm_name
                params_form.empty()
                
        


    # Logic for training a model
    if train_model_bool == 'Yes' and 'Yes' not in [params_bool,transform_data_bool,predict_bool]:
        user_data_check()
        check_cache_data()
        model, _ = check_cache_hyperparams()
        trained_model,X_test,y_test = train_model(data_key,data_source,model)
        st.session_state.user_session_data[ctx.session_id]['model'] = trained_model

        st.title("Model Accuracy")
        if prediction_task == 'Classification':
            report = metrics.classification_report(trained_model.predict(X_test),y_test,output_dict=True)
            st.table(pd.DataFrame(report).T)
            
        else:
            st.text(trained_model.score(X_test,y_test))


    # perform data transformations
    if transform_data_bool == 'Yes' and 'Yes' not in [params_bool,train_model_bool,predict_bool]:
            user_data_check()
            check_cache_data()

            X_train,X_test,y_train,y_test = st.session_state.user_session_data[ctx.session_id]['data']

            try:
                st.table(X_train.describe())
            except:
                st.table(pd.DataFrame(X_train,columns=st.session_state.user_session_data[ctx.session_id]['feature_names'][0:-1]).describe())


            data_transformation_form = st.empty()
            with data_transformation_form.form("transformData"):
                transform_x = st.selectbox("Independent Variable Transformations",['None','StandardScaler','MinMaxScaler','Log-Transform'])
                transform_y = st.selectbox("Dependent Variable Transformations",['None','Log-Transform'])
                submitted = st.form_submit_button("submit")
            
            if submitted:
                if transform_x != 'None' and transform_x != 'Log-Transform':
                    module_name = data_preprocess_dict[transform_x]["module_name"]
                    object_class = data_preprocess_dict[transform_x]["class"]
                    data_preprocessor = instantiate_obj(object_class,module_name)
                    X_train = data_preprocessor.fit_transform(X_train)
                    X_test = data_preprocessor.fit_transform(X_test)
                    st.success(f"X_train Shape:{X_train.shape}, X_test Shape:{X_test.shape}")

                
                if transform_x == 'Log-Transform':
                    X_train = np.log(X_train)
                    X_test = np.log(X_test)
                    st.success(f"X_train Shape:{X_train.shape}, X_test Shape:{X_test.shape}")


                
                if transform_y == 'Log-Transform':
                    y_train = np.log(y_train)
                    y_test = np.log(y_test)
                    st.success(f"Y_train Shape:{y_train.shape}, Y_test Shape:{y_test.shape}")
                st.session_state.user_session_data[ctx.session_id]['data'] = [X_train,X_test,y_train,y_test]

                data_transformation_form.empty()



    if predict_bool == 'Yes' and 'Yes' not in [params_bool,transform_data_bool,train_model_bool]:
        
        user_data_check()
        check_cache_data()
        model,model_param_dict = check_cache_hyperparams()


        prediction_form = st.empty()
        preDict = {key:'' for key in st.session_state.user_session_data[ctx.session_id]['data_cols'][0:-1]}

        with prediction_form.form('predict_here'):
            for key,value in preDict.items():
                preDict[key] = st.text_input(f"{key}",model_param_dict.get(key,value))
            submitted = st.form_submit_button("submit")
            
        if submitted:
          trained_model,X_test,y_test = train_model(data_key,data_source,model)
          sample_data = np.array([int(feature) for key,feature in preDict.items()])
          sample_data = sample_data.reshape(1,len(sample_data))
          st.session_state.user_session_data[ctx.session_id]['pred_algorithm_name'] = algorithm_name
          st.success(f'{trained_model.predict(sample_data)}')
          prediction_form.empty() 
          


    
    return None



main()
