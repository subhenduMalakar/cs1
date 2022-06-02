from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score , log_loss
xgb.set_config(verbosity=0)
import opendatasets as od
from flask import Flask
from flask_caching import Cache
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'SimpleCache' 
cache = Cache(app)


@app.before_first_request
def do_something_only_once():
    # global feature_min, orders_d, hours_order_rate,order_rate_in_different_days,u_order_rate_in_different_hours,u_order_rate_in_different_days,xgb_model
    od.download('https://www.kaggle.com/datasets/subhendumalakar/cs1-ai')
    feature_min = joblib.load('cs1-ai/feature_min_n.joblib') 
    orders_d = joblib.load('cs1-ai/orders_d.joblib')
    tu=joblib.load('cs1-ai/other_f.joblib') #hours_order_rate,order_rate_in_different_days,u_order_rate_in_different_hours,u_order_rate_in_different_days
    xgb_model = joblib.load('cs1-ai/xgb_model_xgb_final.joblib')

    cache.set("feature_min", feature_min)
    cache.set("tu", tu)
    cache.set("xgb_model", xgb_model)
    cache.set("orders_d", orders_d)


sched = BackgroundScheduler(daemon=True)
sched.add_job(do_something_only_once,'interval',minutes=10)
sched.start()


@app.route("/",methods=['GET'])
def hello():
    return render_template('index.html') 

@app.route("/",methods=['POST'])
def get_pre():
    user = request.form.get('user_id')
    r=str(ready(int(user))).split(" ")
    products=pd.read_csv('products.csv')
    products_name=[]
    for i in r:
        products_name.append(np.array(products[products['product_id']==int(i)]['product_name'])[0])
    if len(products_name)==0:
        render_template('index.html',result="Try other User Id's") 
    return render_template('index.html',result=products_name) 

def ready(user):
    hours_order_rate,order_rate_in_different_days,u_order_rate_in_different_hours,u_order_rate_in_different_days=cache.get('tu')
    orders_d=cache.get('orders_d')
    feature_min=cache.get('feature_min')
    xgb_model=cache.get('xgb_model')

    orders_d=orders_d[orders_d['user_id']==user]
    feature_min=feature_min[feature_min['user_id']==user]

    def prepare_test_data(features_u_p_c):
        global marged_t
        tmp=orders_d
        
        tmp=tmp[['user_id','order_id','order_number','order_dow','order_hour_of_day','days_since_prior_order']]

        marged_t=pd.merge(tmp,features_u_p_c, how='left', on=['user_id'])

        #Other features
        marged_t=pd.merge(marged_t,hours_order_rate,on=['product_id','order_hour_of_day'],how='left') 
        marged_t=pd.merge(marged_t,order_rate_in_different_days,on=['product_id','order_dow'],how='left') 

        marged_t=pd.merge(marged_t,u_order_rate_in_different_hours,on=['user_id','order_hour_of_day'],how='left') 
        marged_t=pd.merge(marged_t,u_order_rate_in_different_days,on=['user_id','order_dow'],how='left') 

        marged_t=marged_t.dropna()

        marged_t['days_since_prior_order']=marged_t['days_since_prior_order'].astype(np.float16)
        marged_t['product_id']=marged_t['product_id'].astype(np.int64)
        marged_t['order_id']=marged_t['order_id'].astype(np.int64)

    prepare_test_data(feature_min) #Get marged_t

    feature_names=['total_order_product_order_ratio',
                        'number_of_orders_since_last_order',
                        'last_5_u_p_reorder_p',
                        'last_5_u_p_reorder',
                        'total_unique_product_order_by_u',
                        'user_reorder_rate',
                        'total_product_by_u',
                        'median_days_bwt_orders',
                        'median_products_in_cart',
                        'median_order_dow',
                        'median_order_hour_of_day',
                        'Median_days_interval',
                        'user_reorder_of_Unique_product',
                        'u_days_since_prior_order',
                        'median_position_in_cart_p',
                        'p_Unique_number_of_user_o',
                        'product_reorder_rate',
                        'p_median_days_bwt_orders',
                        'p_Median_days_interval',
                        'order_number',
                        'order_dow',
                        'order_hour_of_day',
                        'days_since_prior_order',
                        'order_rate_in_different_hours',
                        'order_rate_in_different_days',
                        'u_order_rate_in_different_hours',
                        'u_order_rate_in_different_days']
    def get_submissions(xgb_model,TD,marged_t):
        discard_fields = ["product_id","order_id", "user_id"]
        marged_t=marged_t[["product_id","order_id","user_id"]+feature_names]
        X = xgb.DMatrix(marged_t.drop(discard_fields, axis=1))

        s = xgb_model.predict(X)
        # print(s,'######################################################')
        # print(marged_t)
        # print(s,'######################################################')
        marged_t["reordered"]=s
        marged_t["reordered"] = np.where(marged_t.reordered > TD, 1, 0)

        ss = marged_t[marged_t.reordered == 1]
        ss_Final=ss.groupby("user_id")['product_id'].apply(lambda x: ' '.join([str(e) for e in set(x)])).reset_index(name='products')
        ss_Final.reset_index(inplace=True)
        return ss_Final['products'][0]
    TD=0.5
    ss_Final=get_submissions(xgb_model,TD, marged_t)
    if len(ss_Final)==0:
        return 0
    return ss_Final

# if __name__=='__main__':
#     app.run(port=8080,debug=True)