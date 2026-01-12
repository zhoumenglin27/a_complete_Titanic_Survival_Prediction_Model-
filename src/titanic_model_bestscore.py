import pandas as pd 
import numpy as np 
tfilename=r"train.csv"
ffilename=r"test.csv"#读取数据
train_data=pd.read_csv(tfilename)
test_data=pd.read_csv(ffilename)

train_data_copy=train_data.copy()
train_data_copy_target=train_data_copy[['Survived']]#目标值
train_data_copy=train_data_copy.drop(columns=['PassengerId','Survived'])#数据集
columns_num=train_data_copy.columns
#1.填充数字类型的缺失值
from sklearn.impute import SimpleImputer
def fill_null(train_data):
	train_data=train_data[columns_num]
	train_data_num=train_data.select_dtypes(include=[np.number])
	if train_data_num.isna().any().any():
		num_imputer=SimpleImputer(strategy='median')
		fill_numm_result=num_imputer.fit_transform(train_data_num)
		train_data[train_data_num.columns]=fill_numm_result
	train_data_text = train_data.select_dtypes(include=['object', 'category'])
	if train_data_text.isna().any().any():
		text_imputer = SimpleImputer(strategy='constant', fill_value='missing')
		fill_text_result = text_imputer.fit_transform(train_data_text)
		train_data[train_data_text.columns] = fill_text_result
	return train_data

#2.对年龄进行分箱
def age_binning_titanic(train_data):
    bins = [0, 12, 18, 25, 35, 50, 65, 100]
    labels = ['婴儿儿童', '青少年', '青年', '青壮年', '中年', '中老年', '老年']
    train_data['Age_Bin'] = pd.cut(train_data['Age'], bins=bins, labels=labels, right=False)
    train_data=train_data.drop(columns='Age')
    return train_data

#3.求车票单价
def single_fare(train_data):
	ticket_counts = train_data[['Ticket']].value_counts().reset_index(name='tic_num')	
	train_data=pd.merge(train_data,ticket_counts,how='left',left_on='Ticket',right_on='Ticket',suffixes=('', '_right'))
	train_data['single_fare']=train_data['Fare']/train_data['tic_num']
	train_data=train_data.drop(columns=['Fare','tic_num'])
	return train_data

#4.删除车票
def drop_tickets(train_data):
	train_data=train_data.drop(columns='Ticket')
	return train_data

#5.删除车厢
def drop_cabin(train_data):
	train_data=train_data.drop(columns='Cabin')
	return train_data

#6.拆分名字
def split_name(train_data):
	split_name1=train_data['Name'].str.split(',',n=1,expand=True)
	split_name2=split_name1[1].str.split('.',n=1,expand=True)
	train_data['title_name']=split_name2[0]
	train_data=train_data.drop(columns=['Name'])
	return train_data

#7.下面进行数据独热编码，处理Pclass、Sex、Embarked、Age_Bin、Cabin_num、title_name
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
code_list=['Pclass','Sex','Embarked','Age_Bin','title_name']
onehot_codes=OneHotEncoder(handle_unknown='infrequent_if_exist',sparse_output=False)
preprocessor = ColumnTransformer(transformers=[('cat', onehot_codes, code_list)],
    remainder='passthrough'  # 保留其他列（如SibSp, Parch, single_fare等）
)

	
#11.对票价进行处理
def scale_fare(train_data):
	train_data['single_fare']=np.log(train_data['single_fare']+1)
	return train_data

#12.建一个超参数调优过程
from sklearn.model_selection import GridSearchCV


#13.做一个原始数据清洗的pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

data_fill_null=FunctionTransformer(fill_null)
data_age_binning_titanic=FunctionTransformer(age_binning_titanic)
data_single_fare=FunctionTransformer(single_fare)
data_drop_tickets=FunctionTransformer(drop_tickets)
data_drop_cabin=FunctionTransformer(drop_cabin)
data_split_name=FunctionTransformer(split_name)
data_scale_fare=FunctionTransformer(scale_fare)

data_cleaning=make_pipeline(
	data_fill_null,
	data_age_binning_titanic,
	data_single_fare,
	data_drop_tickets,
	data_drop_cabin,
	data_split_name,
	data_scale_fare
	)
clean_data=data_cleaning.fit_transform(train_data_copy)

#14.做模型预测的pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
model_one=Pipeline([
	('preprocessor', preprocessor),
	#('feature_selection', SelectKBest(score_func=chi2, k=10)),  # 卡方选择)
	 ('KNN',KNeighborsClassifier())])
KNN_search_grid=[{'KNN__n_neighbors':[3,5,10],'KNN__weights':['uniform','distance'],'KNN__algorithm':['brute']},#超参数调优过程
                 {'KNN__n_neighbors':[3,5,10],'KNN__weights':['uniform','distance'],'KNN__algorithm':['ball_tree','kd_tree'],'KNN__leaf_size':[10,20,30,40,50]}]
GridSearch=GridSearchCV(estimator=model_one,param_grid=KNN_search_grid,scoring='roc_auc',n_jobs=-1,refit=True,cv=5)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(clean_data, train_data_copy_target, test_size=0.2, random_state=42)

GridSearch.fit(X_train,y_train)
print(GridSearch.best_estimator_)
print(GridSearch.best_params_)
print(GridSearch.best_score_)
print(GridSearch.score(X_test, y_test))

#换个模型
from sklearn.linear_model import LogisticRegressionCV
model_two=Pipeline([
	('preprocessor', preprocessor),
	('scaler', MinMaxScaler(feature_range=(0.001, 1))),
	#('feature_selection', SelectKBest(score_func=chi2, k=10)),  # 卡方选择)
	('LR',LogisticRegressionCV())])
LR_search_grid=[{'LR__Cs':[0.1,1,10,1000],'LR__l1_ratios':[[0,1], None],'LR__cv':[5,10],'LR__penalty':['l1', 'l2', 'elasticnet']}#超参数调优过程
                ]
GridSearch=GridSearchCV(estimator=model_two,param_grid=LR_search_grid,scoring='roc_auc',n_jobs=-1,refit=True,cv=5)
GridSearch.fit(X_train,y_train)

print(GridSearch.best_estimator_)
print(GridSearch.best_params_)
print(GridSearch.best_score_)
print(GridSearch.score(X_test, y_test))



t_test_data=test_data.drop(columns=['PassengerId'])
t_clean_data=data_cleaning.transform(t_test_data)

results=GridSearch.predict(t_clean_data)


assert len(results) == len(test_data), "预测结果数量与测试集不匹配"

# 创建提交DataFrame
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': results  # 直接使用你的数组
})

# 保存为CSV
submission.to_csv('submission.csv', index=False)#上传后成绩为0.785