import logging
import sys
from tqdm import tqdm
import time 


def ML_run():
    logger.info("Start runing...")
    logger.info("Importing...")
    import warnings
    warnings.filterwarnings('ignore')
    import pandas as pd
    import numpy as np
    np.random.seed(1)
    import sklearn
    import sklearn.model_selection
    import sklearn.metrics
    import sklearn.ensemble
    import seaborn as sns
    sns.set()
    import matplotlib.pyplot as plt
    from matplotlib import rc
    # %matplotlib inline

    #Result out
    file_all_out = f"{bacteria}_{antb}_Complete_Results_{validation_no}-fold_CV.csv"
    logger.info("Start Reading Data.")
    data = pd.read_csv(file_name)
    logger.info("Reading Data Done.")

    #Split data into features and labels
    X = data.iloc[:, 1:-1] 
    Y = data.iloc[:,-1] # last column label

    #Label size and matrix size
    All_Set_Data_size = data.groupby(antb).size()
    All_Set_Matrix_size = data.shape

    logger.info("Start import classifiers...")
    #Import classifiers
    from sklearn import model_selection
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import f1_score
    from sklearn.metrics import auc
    from sklearn.metrics import precision_score, recall_score
    import pickle
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import LeaveOneOut 
    from sklearn.model_selection import cross_val_score
    logger.info("Import classifiers Done.")
    #Create dataframes for outputs
    Training_Performance = pd.DataFrame(columns=[])
    Test_Performance = pd.DataFrame(columns=[])
    Tf_CV = pd.DataFrame(columns=[])
    Area_Under_ROC = pd.DataFrame(columns=[])
    Area_Under_Precision_Recall = pd.DataFrame(columns=[])
    Model_Predict = pd.DataFrame(columns=[])

    # Split data into 6 equal parts
    skf = StratifiedKFold(n_splits=validation_no, random_state=42, shuffle=True)
    i = 0
    for train_index, test_index in tqdm(skf.split(X, Y), desc="StratifiedKFold"):
        logger.info("Start StratifiedKFold ind: {}.".format(i))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        #Build and evaluate models
        models = []
        models.append(('LogR', LogisticRegression()))
        models.append(('gNB', GaussianNB()))
        models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
        models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
        models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
        models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('mNB', MultinomialNB()))
        models.append(('ABC', AdaBoostClassifier()))
        models.append(('GBC', GradientBoostingClassifier()))
        models.append(('ETC', ExtraTreesClassifier()))
        models.append(('BC', BaggingClassifier()))
        logger.info("Build models done.")

 

        #Training performances
        myDF1 = pd.DataFrame(columns=[])
                #Test performances
        myDF2 = pd.DataFrame(columns=[])
        
        myDF3 = pd.DataFrame(columns=[])
        for name, model in tqdm(models, desc="myDF3"):  
            logger.info("Start kfold {}...".format(name))
            kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
            logger.info("10fold model: {} done. ".format(name))
            results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
            mean= results.mean().round(3)
            std = results.std()
            logger.info("Model {} 10fold: mean:{},\t std:{}.".format(name, mean, std))
            myDF3 = myDF3._append({'classifier': name, f'ten_f_CV{i+1}':mean}, ignore_index = True)
        Tf_CV = pd.concat([Tf_CV, myDF3], axis = 1)

        myDF4 = pd.DataFrame(columns=[])
        myDF5 = pd.DataFrame(columns=[])
        for name, model in tqdm(models, desc="myDF12"):  
            logger.info("Start training {}...".format(name))  
            model = model.fit(X_train, Y_train)
            logger.info("Train model: {} done. Then start predicting X_train...".format(name))
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)
            logger.info("{} predict done.".format(name))
            Tr_precision = precision_score(Y_train, Y_train_pred, average='macro').round(3)
            Tr_recall = recall_score(Y_train, Y_train_pred, average='macro').round(3)
            Tr_f1 = f1_score (Y_train, Y_train_pred, average='macro').round(3)
            logger.info("Model {} X_train: Tr_precision:{},\t Tr_recall:{},\t Tr_f1:{}.".format(name, Tr_precision, Tr_recall, Tr_f1))
            myDF1 = myDF1._append({'classifier': name, f'tr_precision{i+1}': Tr_precision, f'tr_recall{i+1}': Tr_recall, f'tr_f1 {i+1}':Tr_f1}, ignore_index = True)
            # report = sklearn.metrics.classification_report(Y_test, Y_test_pred)
            Te_precision = precision_score(Y_test, Y_test_pred, average='macro').round(3)
            Te_recall = recall_score(Y_test, Y_test_pred, average='macro').round(3)
            Te_f1 = f1_score (Y_test, Y_test_pred, average='macro').round(3)
            logger.info("Model {} X_train: Te_precision:{},\t Te_recall:{},\t Te_f1:{}.".format(name, Te_precision, Te_recall, Te_f1))
            myDF2 = myDF2._append({'classifier': name, f'te_precision{i+1}': Te_precision, f'te_recall{i+1}': Te_recall, f'te_f1 {i+1}':Te_f1}, ignore_index = True)
            y_pred_proba = model.predict_proba(X_test)[::,1]
            logger.info("y_pred_proba: {} done. ".format(name))
            fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba, pos_label = None)
            a_u_c = roc_auc_score(Y_test, y_pred_proba).round(3)
            logger.info("Model {} AU_ROC: fpr:{},\t tpr:{}.".format(name, fpr, tpr))
            myDF4 = myDF4._append({'a classifier': name, f'au ROC {i+1}': a_u_c}, ignore_index = True)
            y_pred_proba5 = model.predict_proba(X_test)
            logger.info("y_pred_proba5: {} done. ".format(name))
            # keep probabilities for the positive outcome only
            y_pred_proba5 = y_pred_proba5[:, 1]
            #predict class vlaues
            y_pred5 = model.predict(X_test)
            # calculate precision-recall curve
            logger.info("Predict: {} done. calculate precision-recall curve...".format(name))
            precision, recall, _ = precision_recall_curve(Y_test, y_pred_proba5)
            logger.info("Predict: {} done. calculate au precision-recall curve...".format(name))
            # calculate au precision-recall curve
            area = auc(recall, precision).round(3)
            # calculate f1 score
            logger.info("Predict: {} done. calculate f1 score...".format(name))
            f1 = f1_score(Y_test, y_pred5).round(3)
            logger.info("Model {} AU_ROC: precision:{},\t recall:{},\t area:{},\t f1:{}.".format(name, precision, recall,area,f1))
            myDF5 = myDF5._append({'a classifier': name, f'au PR {i+1}': area}, ignore_index = True)
        Training_Performance = pd.concat([Training_Performance, myDF1], axis = 1)
        Test_Performance = pd.concat([Test_Performance, myDF2], axis = 1)
        Area_Under_ROC = pd.concat([Area_Under_ROC, myDF4], axis = 1)
        Area_Under_Precision_Recall = pd.concat([Area_Under_Precision_Recall, myDF5], axis = 1)
        i += 1

    Tf_CV.to_csv(f'{bacteria}_{antb}_All_Set_Tf_CV_{validation_no}-fold_CV.csv', encoding='utf-8') 
    Training_Performance.to_csv(f'{bacteria}_{antb}_All_Set_Performance_Training_Performance_{validation_no}-fold_CV.csv', encoding='utf-8') 
    Test_Performance.to_csv(f'{bacteria}_{antb}_All_Set_Test_Training_Performance_{validation_no}-fold_CV.csv', encoding='utf-8') 
    Area_Under_ROC.to_csv(f'{bacteria}_{antb}_All_Set_Test_Area_Under_ROC_{validation_no}-fold_CV.csv', encoding='utf-8') 
    Area_Under_Precision_Recall.to_csv(f'{bacteria}_{antb}_All_Set_Test_Area_Under_Precision_Recall_{validation_no}-fold_CV.csv', encoding='utf-8')


    #Model names
    Models = Tf_CV.iloc[:, 0] 
    #Calculating the mean of all folds
    logger.info("Calculating the mean of all folds")
    #training f1 average Training_Performance.filter(like='tr_precision').mean(axis=1).round(3)
    tr_f1_avg = Training_Performance.filter(like='tr_f1').mean(axis=1).round(3)
    tr_f1_avg = tr_f1_avg.rename('tr_f1_avg', inplace=True)
    #Training precision average
    tr_precision_avg = Training_Performance.filter(like='tr_precision').mean(axis=1).round(3)
    tr_precision_avg = tr_precision_avg.rename('tr_precision_avg', inplace=True)

    #Training recall average
    tr_recall_avg = Training_Performance.filter(like='tr_recall').mean(axis=1).round(3)
    tr_recall_avg = tr_recall_avg.rename('tr_recall_avg', inplace=True)

    #Test f1 average
    te_f1_avg = Test_Performance.filter(like='te_f1').mean(axis=1).round(3)
    te_f1_avg = te_f1_avg.rename('te_f1_avg', inplace=True)

    #Test precision average
    te_precision_avg = Test_Performance.filter(like='te_precision').mean(axis=1).round(3)
    te_precision_avg = te_precision_avg.rename('te_precision_avg', inplace=True)

    #Test recall average
    te_recall_avg = Test_Performance.filter(like='te_recall').mean(axis=1).round(3)
    te_recall_avg = te_recall_avg.rename('te_recall_avg', inplace=True)

    #Tf_CV average
    Tf_CV_Avg = Tf_CV.filter(like='ten_f_CV').mean(axis=1).round(3)
    Tf_CV_Avg = Tf_CV_Avg.rename('Tf_CV_Avg', inplace=True)

    #Area_Under_ROC average
    au_ROC_avg = Area_Under_ROC.filter(like='au ROC').mean(axis=1).round(3)
    au_ROC_avg = au_ROC_avg.rename('au_ROC_avg', inplace=True)

    #Area_Under_Precision_Recall average
    au_PR_avg= Area_Under_Precision_Recall.filter(like='au PR').mean(axis=1).round(3)
    au_PR_avg = au_PR_avg.rename('au_PR_avg', inplace=True)

    #Accumulating all dataframes
    logger.info("Accumulating all dataframes")
    frames2 = [Models, tr_precision_avg, tr_recall_avg, tr_f1_avg, te_precision_avg, te_recall_avg, te_f1_avg, Tf_CV_Avg, au_ROC_avg, au_PR_avg]

    #Result all set
    Final_All_set_Results= pd.concat(frames2, axis=1)
    
    # Result out Modify！！！
    logger.info("Result all set")
    with open (file_all_out, 'a+') as rf:
        rf.write(f'\n{bacteria} {antb} All set Results\n')
        rf.write(f'\n{All_Set_Data_size}\n')
        rf.write(f'\nmatrix_size: {All_Set_Matrix_size}\n\n')

        Final_All_set_Results.to_csv(rf)

    #Fit on whole set and predict the labels
    models1 = []
    models1.append(('LogR', LogisticRegression()))
    models1.append(('gNB', GaussianNB()))
    models1.append(('SVM', SVC(kernel = 'rbf', probability=True)))
    models1.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
    models1.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
    models1.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
    models1.append(('LDA', LinearDiscriminantAnalysis()))
    models1.append(('mNB', MultinomialNB()))
    models1.append(('ABC', AdaBoostClassifier()))
    models1.append(('GBC', GradientBoostingClassifier()))
    models1.append(('ETC', ExtraTreesClassifier()))
    models1.append(('BC', BaggingClassifier()))

    #Predict Lables
    logger.info("Predict Lables")
    predict_lab = []
    predicted_df = pd.DataFrame(columns=[])

    for name, model in models1:
        logger.info("model pickle_files {}...".format(name))
    #Fit the model on whole dataset
        model.fit(X, Y)
        pickle_files = f'{antb}_{name}.sav'

        #Save the mode as pickle file
        pickle.dump(model, open(pickle_files, 'wb'))

        #Load the model from disk
        logger.info("load model{}...".format(name))
        loaded_model = pickle.load(open(pickle_files, 'rb'))

        #Predicting the new data
        pr = pd.read_csv(To_predict)
        Xnew = pr.iloc[:, 1:-1]

        ynew = loaded_model.predict(Xnew)
        predict_lab.append({'predicted label':ynew})
        labels = pd.DataFrame(data=ynew, columns=[f'prediction by {name}'])
        predicted_df = pd.concat([predicted_df, labels], axis=1)

    #Labels predicted by each model    
    Model_Predict = pd.DataFrame(predict_lab)

    #Separate df for each isolate and predicted label by models
    Predicted_Labels = pd.concat([pr['Isolate'], predicted_df], axis=1)  

    #Leave one out cross validation
    #List for output
    Loo = []

    #Leave one out validation
    cv = LeaveOneOut()

    #Accumulate models  
    models=[]
    models.append(('LogR', LogisticRegression()))
    models.append(('gNB', GaussianNB()))
    models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
    models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
    models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('mNB', MultinomialNB()))
    models.append(('ABC', AdaBoostClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('ETC', ExtraTreesClassifier()))
    models.append(('BC', BaggingClassifier()))

    #Evaluate each model
    for name, model in models:
        logger.info("Evaluating model {}...".format(name))
        # print("Evaluating model {}...".format(name))
        # fit model
        scores = cross_val_score(model, X, Y, cv = cv, scoring='accuracy', n_jobs=8).mean().round(3)
        Loo.append({'Loo_CV': scores})

    Loo_CV = pd.DataFrame(Loo)
    Final_All_set_Results = pd.concat([Final_All_set_Results, Loo_CV], axis=1)
        #Export results separately
    Final_All_set_Results.to_csv(f'{bacteria}_{antb}_All_Set_Performance_{validation_no}-fold_CV.csv', encoding='utf-8')
    #print(f'All Set Results {antb} {bacteria}')
    #display(Final_All_set_Results)

    #Result out Modify！！！
    with open (file_all_out, 'a+') as rf:
        rf.write('\nPredicted Labels All Set\n')
        Predicted_Labels.to_csv(rf)


    #Selecting important features in each fold from tree based-classifiers
    clfk = ExtraTreesClassifier(random_state=1)

    #Dataframes for output
    feat_Df = pd.DataFrame(columns=[])
    scores = []
    test_scores = []
    check_feat = []
    Output = pd.DataFrame()

    #Split the data
    skf = StratifiedKFold(n_splits=validation_no, random_state=42, shuffle=True)
    j = 0
    for train_index, test_index in tqdm(skf.split(X, Y), desc="StratifiedKFold"):
        logger.info("model pickle_files {},{}...".format(train_index,test_index ))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        #Fit the model
        logger.info("clfk.fit train model {}...".format(train_index))
        modelk = clfk.fit(X_train,Y_train)
        predictions = clfk.predict(X_test)
        logger.info("predictions done")
        scores.append(clfk.score(X_test, Y_test))
        feat = clfk.feature_importances_

        #Select the column header from first to the second last
        logger.info("Select the column header from first to the second last")
        colhead = list(np.array([data.columns[1:-1]]).T)

        #Zip two columns into a dataframe
        logger.info("Zip two columns into a dataframe")
        list_of_tuplesk= list(zip(colhead, feat))

        #Create features dataframe
        logger.info("Create features dataframe Feature fold {}...".format(j))
        feature_importancek = pd.DataFrame(list_of_tuplesk, columns = [f'Feature fold{j}', f'Importance fold{j}'])

        #Sort the dataframe, descending
        logger.info("Sort the dataframe Feature fold {}...".format(j))
        feature_importance_sortedk = feature_importancek.sort_values(by=f'Importance fold{j}', ascending=False)

        #Remove the square brackets from the dataframe
        feature_importance_sortedk [f'Feature fold{j}'] = feature_importance_sortedk[f'Feature fold{j}'].str.get(0)
        feature_importance_sortedk = feature_importance_sortedk.round(3)

        #Sort the features
        feat_sort_df = pd.DataFrame(feature_importance_sortedk)
        feat_sort_df.reset_index(drop=True, inplace=True)
        feat_Df.reset_index(drop=True, inplace=True)
        feat_Df = pd.concat([feat_Df, feat_sort_df], axis= 1)
        j += 1

    #Select the top genes out from range
    logger.info("Select the top genes out from range")
    top_genes_range = 100

    #Make dataframe of selected top dataframes
    Top_consistent = feat_Df.iloc[0:top_genes_range, :]

    #Separate each column to separate dataframe and find common in all
    logger.info("Separate each column to separate dataframe and find common in all")
    cdf1 = Top_consistent[['Feature fold0']].rename(columns={"Feature fold0": "Feature"})
    cdf2 = Top_consistent[['Feature fold1']].rename(columns={"Feature fold1": "Feature"})
    cdf3 = Top_consistent[['Feature fold2']].rename(columns={"Feature fold2": "Feature"})
    cdf4 = Top_consistent[['Feature fold3']].rename(columns={"Feature fold3": "Feature"})
    cdf5 = Top_consistent[['Feature fold4']].rename(columns={"Feature fold4": "Feature"})
    cdf6 = Top_consistent[['Feature fold5']].rename(columns={"Feature fold5": "Feature"})

    #Merging common in all folds
    logger.info("Merging common in all folds")
    merge12 = pd.merge(cdf1, cdf2, how='inner', on=['Feature'])
    merge123 = pd.merge(merge12, cdf3, how='inner', on=['Feature'])
    merge1234 = pd.merge(merge123, cdf4, how='inner', on=['Feature'])
    merge12345 = pd.merge(merge1234, cdf5, how='inner', on=['Feature'])
    Consistent_Genes_per_fold = pd.merge(merge12345, cdf6, how='inner', on=['Feature'])
    Final_Consistent_Genes_per_fold = Consistent_Genes_per_fold.iloc[:50, :]

    #Create a result file
    logger.info("Create a result file")
    with open (file_all_out, 'a+') as rf:
        rf.write(f'\n{bacteria} {antb} Consistent Genes per {validation_no} fold validation All Set\n')
        Final_Consistent_Genes_per_fold.to_csv(rf)

    #Export consistent genes as separate file
    logger.info("Export consistent genes as separate file")
    Final_Consistent_Genes_per_fold.to_csv(f'{bacteria}_{antb}_Consistent_Genes_Per_{validation_no}-fold_CV.csv', encoding='utf-8')


    #INTERSECTION SET RUN
    #Read gene_ast matrix
    logger.info("INTERSECTION SET RUN")
    open_gene_ast = pd.read_csv(file_name)

    #Open consistent genes based per validation
    open_consistent_genes = pd.read_csv(f'{bacteria}_{antb}_Consistent_Genes_Per_{validation_no}-fold_CV.csv')

    #Read antibiotic to predict
    logger.info("Read antibiotic to predict")
    Antibio_to_predict = pd.read_csv(f'{To_predict}')

    #Make separate dataframe with just consistent genes
    target_genesTT = open_consistent_genes[['Feature']].rename(columns={'Feature': 'Consistent genes'})

    #No of top consistent genes
    num = 15
    target_genesTT = target_genesTT.iloc[:num, :]

    #Sort the consistent genes
    logger.info("Sort the consistent genes")
    target_genesTT = target_genesTT.sort_values('Consistent genes')

    #Adding antibiotic lable at the end
    logger.info("Adding antibiotic lable at the end")
    target_genes_good = target_genesTT._append({'Consistent genes': f'{antb}'}, ignore_index=True)

    #Converting consistent genes to a list
    logger.info("Converting consistent genes to a list")
    column_list = target_genes_good['Consistent genes'].tolist()

    #Adding phenotype lable at the end
    logger.info("Adding phenotype lable at the end")
    target_genes_good1 = target_genesTT._append({'Consistent genes': 'phenotype'}, ignore_index=True)

    #Converting consistent genes with phenotype to a list
    logger.info("Converting consistent genes with phenotype to a list")
    column_list1 = target_genes_good1['Consistent genes'].tolist()

    #Make data consisting only with consistent genes 
    logger.info("Make data consisting only with consistent genes ")
    data = open_gene_ast[column_list]

    #Label size and matrix size
    logger.info("Label size and matrix size ")
    Intersection_Data_size = data.groupby(antb).size()
    Intersection_Matrix_size = data.shape

    #Split the data to features and labels
    logger.info("Split the data to features and labels ")
    X = data.iloc[:, 0:-1]
    Y = data.iloc[:,-1]

    #Create dataframes for outputs
    logger.info("Create dataframes for outputs ")
    Training_Performance = pd.DataFrame(columns=[])
    Test_Performance = pd.DataFrame(columns=[])
    Tf_CV = pd.DataFrame(columns=[])
    Area_Under_ROC = pd.DataFrame(columns=[])
    Area_Under_Precision_Recall = pd.DataFrame(columns=[])
    Model_Predict = pd.DataFrame(columns=[])

    #Split data into 6 equal parts
    logger.info("Split data into 6 equal parts")
    skf = StratifiedKFold(n_splits=validation_no, random_state=42, shuffle=True)
    i = 0
    for train_index, test_index in tqdm(skf.split(X, Y), desc="StratifiedKFold"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        #Build and evaluate models
        models = []
        models.append(('LogR', LogisticRegression()))
        models.append(('gNB', GaussianNB()))
        models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
        models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
        models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
        models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('mNB', MultinomialNB()))
        models.append(('ABC', AdaBoostClassifier()))
        models.append(('GBC', GradientBoostingClassifier()))
        models.append(('ETC', ExtraTreesClassifier()))
        models.append(('BC', BaggingClassifier()))


                #Ten-fold cross validation
        myDF3 = pd.DataFrame(columns=[])
        for name, model in tqdm(models, desc="myDF3"):
            logger.info("Start kfold {}...".format(name))
            kfold = model_selection.KFold(n_splits=10, random_state=42, shuffle=True)
            logger.info("10fold model: {} done. ".format(name))
            results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
            mean = results.mean().round(3)
            std = results.std()
            logger.info("Model {} 10fold: mean:{},\t std:{}.".format(name, mean, std))
            myDF3 = myDF3._append({'classifier': name, f'ten_f_CV{i+1}':mean}, ignore_index = True)
        Tf_CV = pd.concat([Tf_CV, myDF3], axis = 1)

        #Training performances
        myDF1 = pd.DataFrame(columns=[])
        myDF2 = pd.DataFrame(columns=[])
        myDF4 = pd.DataFrame(columns=[])
        myDF5 = pd.DataFrame(columns=[])
        for name, model in tqdm(models, desc="myDF1245"): 
            logger.info("Start training {}...".format(name))    
            model = model.fit(X_train, Y_train)
            logger.info("Train model: {} done. Then start predicting X_train...".format(name))
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)
            logger.info("{} predict done.".format(name))
            Tr_precision = precision_score(Y_train, Y_train_pred, average='macro').round(3)
            Tr_recall = recall_score(Y_train, Y_train_pred, average='macro').round(3)
            Tr_f1 = f1_score (Y_train, Y_train_pred, average='macro').round(3)
            logger.info("Model {} X_train: Tr_precision:{},\t Tr_recall:{},\t Tr_f1:{}.".format(name, Tr_precision, Tr_recall, Tr_f1))
            myDF1 = myDF1._append({'classifier': name, f'tr_precision{i+1}': Tr_precision, f'tr_recall{i+1}': Tr_recall, f'tr_f1 {i+1}':Tr_f1}, ignore_index = True)
            # report = sklearn.metrics.classification_report(Y_test, Y_test_pred)
            Te_precision = precision_score(Y_test, Y_test_pred, average='macro').round(3)
            Te_recall = recall_score(Y_test, Y_test_pred, average='macro').round(3)
            Te_f1 = f1_score (Y_test, Y_test_pred, average='macro').round(3)
            logger.info("Model {} X_train: Te_precision:{},\t Te_recall:{},\t Te_f1:{}.".format(name, Te_precision, Te_recall, Te_f1))
            myDF2 = myDF2._append({'classifier': name, f'te_precision{i+1}': Te_precision, f'te_recall{i+1}': Te_recall, f'te_f1 {i+1}':Te_f1}, ignore_index = True)
            y_pred_proba = model.predict_proba(X_test)[::,1]
            logger.info("y_pred_proba: {} done. ".format(name))
            fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba, pos_label = None)
            a_u_c = roc_auc_score(Y_test, y_pred_proba).round(3)
            logger.info("Model {} AU_ROC: fpr:{},\t tpr:{}.".format(name, fpr, tpr))
            myDF4 = myDF4._append({'a classifier': name, f'au ROC {i+1}': a_u_c}, ignore_index = True)
            logger.info("Start predict probabilities {}...".format(name))
            #predict probabilities
            y_pred_proba5 = model.predict_proba(X_test)
            logger.info("y_pred_proba: {} done. ".format(name))
            # keep probabilities for the positive outcome only
            y_pred_proba5 = y_pred_proba5[:, 1]
            #predict class vlaues
            y_pred5 = model.predict(X_test)
            # calculate precision-recall curve
            logger.info("Predict: {} done. calculate precision-recall curve...".format(name))
            precision, recall, _ = precision_recall_curve(Y_test, y_pred_proba5)
            logger.info("Predict: {} done. calculate au precision-recall curve...".format(name))
            # calculate au precision-recall curve
            area = auc(recall, precision).round(3)
            # calculate f1 score
            logger.info("Predict: {} done. calculate f1 score...".format(name))
            f1 = f1_score(Y_test, y_pred5).round(3)
            logger.info("Model {} AU_ROC: precision:{},\t recall:{},\t area:{},\t f1:{}.".format(name, precision, recall,area,f1))
            myDF5 = myDF5._append({'a classifier': name, f'au PR {i+1}': area}, ignore_index = True)

         
        Training_Performance = pd.concat([Training_Performance, myDF1], axis = 1)
        Test_Performance = pd.concat([Test_Performance, myDF2], axis = 1)
        Area_Under_ROC = pd.concat([Area_Under_ROC, myDF4], axis = 1)
        Area_Under_Precision_Recall = pd.concat([Area_Under_Precision_Recall, myDF5], axis = 1)
        i += 1

    Tf_CV.to_csv(f'{bacteria}_{antb}_Intersect_Set_Tf_CV_{validation_no}-fold_CV.csv', encoding='utf-8') 
    Training_Performance.to_csv(f'{bacteria}_{antb}_Intersect_Set_Performance_Training_Performance_{validation_no}-fold_CV.csv', encoding='utf-8') 
    Test_Performance.to_csv(f'{bacteria}_{antb}_Intersect_Set_Performance_Test_Performance_{validation_no}-fold_CV.csv', encoding='utf-8') 
    Area_Under_ROC.to_csv(f'{bacteria}_{antb}_Intersect_Set_Performance_Area_Under_ROC_{validation_no}-fold_CV.csv', encoding='utf-8') 
    Area_Under_Precision_Recall.to_csv(f'{bacteria}_{antb}_Intersect_Set_Performance_Area_Under_Precision_Recall_{validation_no}-fold_CV.csv', encoding='utf-8')
    
    #Model names
    Models = Tf_CV.iloc[:, 0] 

    #Calculating the mean of all folds
    logger.info("Calculating the mean of all folds")
    #training f1 average
    tr_f1_avg = Training_Performance.filter(like='tr_f1').mean(axis=1).round(3)
    tr_f1_avg = tr_f1_avg.rename('tr_f1_avg', inplace=True)

    #Training precision average
    tr_precision_avg = Training_Performance.filter(like='tr_precision').mean(axis=1).round(3)
    tr_precision_avg = tr_precision_avg.rename('tr_precision_avg', inplace=True)

    #Training recall average
    tr_recall_avg = Training_Performance.filter(like='tr_recall').mean(axis=1).round(3)
    tr_recall_avg = tr_recall_avg.rename('tr_recall_avg', inplace=True)

    #Test f1 average
    te_f1_avg = Test_Performance.filter(like='te_f1').mean(axis=1).round(3)
    te_f1_avg = te_f1_avg.rename('te_f1_avg', inplace=True)

    #Test precision average
    te_precision_avg = Test_Performance.filter(like='te_precision').mean(axis=1).round(3)
    te_precision_avg = te_precision_avg.rename('te_precision_avg', inplace=True)

    #Test recall average
    te_recall_avg = Test_Performance.filter(like='te_recall').mean(axis=1).round(3)
    te_recall_avg = te_recall_avg.rename('te_recall_avg', inplace=True)

    #Tf_CV average
    Tf_CV_Avg = Tf_CV.filter(like='ten_f_CV').mean(axis=1).round(3)
    Tf_CV_Avg = Tf_CV_Avg.rename('Tf_CV_Avg', inplace=True)

    #Area_Under_ROC average
    au_ROC_avg = Area_Under_ROC.filter(like='au ROC').mean(axis=1).round(3)
    au_ROC_avg = au_ROC_avg.rename('au_ROC_avg', inplace=True)

    #Area_Under_Precision_Recall average
    au_PR_avg = Area_Under_Precision_Recall.filter(like='au PR').mean(axis=1).round(3)
    au_PR_avg = au_PR_avg.rename('au_PR_avg', inplace=True)

    #Accumulating all dataframes
    logger.info("Accumulating all dataframes")
    frames2 = [Models, tr_precision_avg, tr_recall_avg, tr_f1_avg, te_precision_avg, te_recall_avg, te_f1_avg, Tf_CV_Avg, au_ROC_avg, au_PR_avg]

    Intersection_set_Results= pd.concat(frames2, axis=1)

    logger.info("export Intersection_set_Results separately")
    Intersection_set_Results.to_csv(f'{bacteria}_{antb}_Intersection_Set_Performance_{validation_no}-fold_CV_1.csv', encoding='utf-8')

    #Leave one out cross validation
    logger.info("Leave one out cross validation")
    #List for output
    Loo = []

    #Leave one out validation
    cv = LeaveOneOut()

    #Accumulate models  
    models = []
    models.append(('LogR', LogisticRegression()))
    models.append(('gNB', GaussianNB()))
    models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
    models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
    models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('mNB', MultinomialNB()))
    models.append(('ABC', AdaBoostClassifier()))
    models.append(('GBC', GradientBoostingClassifier()))
    models.append(('ETC', ExtraTreesClassifier()))
    models.append(('BC', BaggingClassifier()))

    #Evaluate each model
    for name, model in models:
        logger.info("Evaluating model {}...".format(name))
        # print("Evaluating model {}...".format(name))
        # fit model
        scores = cross_val_score(model, X, Y, cv = cv, scoring='accuracy', n_jobs=8).mean().round(3)
        Loo.append({'Loo_CV': scores})

    Loo_CV = pd.DataFrame(Loo)
    logger.info("Intersection_set_Results")
    Intersection_set_Results = pd.concat([Intersection_set_Results, Loo_CV], axis=1)

    # print(f'Intersection Set Results {antb} {bacteria}')
    # display(Intersection_set_Results)
    #Results out
    with open (file_all_out, 'a+') as rf:
        rf.write(f'\n{bacteria} {antb} Intersection Set Results\n')

        Intersection_set_Results.to_csv(rf)

        rf.write('\nPredicted Labels Intersection Set\n')
        Predicted_Labels.to_csv(rf)

    # export result separately
    logger.info("export Intersection_set_Results separately")
    Intersection_set_Results.to_csv(f'{bacteria}_{antb}_Intersection_Set_Performance_{validation_no}-fold_CV.csv', encoding='utf-8')



    #RANDOM SET RUN
    logger.info("RANDOM SET RUN")
    #No of genes to shuffle
    num = 16
    logger.info("Read gene_ast matrix")
    #Read gene_ast matrix
    open_gene_ast = pd.read_csv(file_name)
    label = open_gene_ast[[antb]]
    daa = open_gene_ast.drop(['Isolate', antb], axis=1)

    #Create a dataframe for the final output of the program
    logger.info("Create a dataframe for the final output of the program")
    Random_Set_Results = pd.DataFrame(columns=[])
    Loo_CV = pd.DataFrame(columns=[])

    #Select 10 random sets
    logger.info("Select 10 random sets")
    for i in range(10):
        samp = daa.sample(n=num, replace = True, axis=1)
        data = pd.concat([samp, label], axis=1)
        Final_Randon_data_size = data.groupby(antb).size()
        X = data.iloc[:, 0:num]
        Y = data.iloc[:,-1]

        #Dataframes for results
        Tf_CV = pd.DataFrame(columns=[])
        Training_Performance = pd.DataFrame(columns=[])
        Test_Performance = pd.DataFrame(columns=[])
        Area_Under_ROC = pd.DataFrame(columns=[])
        Area_Under_Precision_Recall = pd.DataFrame(columns=[])
        Model_Predict = pd.DataFrame(columns=[])
        skf = StratifiedKFold(n_splits=validation_no, random_state=42, shuffle=True)
        ij = 0

        #Split the data
        logger.info("Split the data")
        for train_index, test_index in tqdm(skf.split(X, Y), desc="StratifiedKFold"):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            #Build model and evaluate models
            models = []
            models.append(('LogR', LogisticRegression()))
            models.append(('gNB', GaussianNB()))
            models.append(('SVM', SVC(kernel = 'rbf', probability=True)))
            models.append(('DT', DecisionTreeClassifier(criterion = 'entropy', random_state=1)))
            models.append(('RF', RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state=0)))
            models.append(('KNN', KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)))
            models.append(('LDA', LinearDiscriminantAnalysis()))
            models.append(('mNB', MultinomialNB()))
            models.append(('ABC', AdaBoostClassifier()))
            models.append(('GBC', GradientBoostingClassifier()))
            models.append(('ETC', ExtraTreesClassifier()))
            models.append(('BC', BaggingClassifier()))
            
            #Training performance

            #Ten-fold cross validation
            myDF = pd.DataFrame(columns=[])
            for name, model in tqdm(models, desc="myDF"): 
                logger.info("Start kfold {}...".format(name))
                kfold = model_selection.StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
                logger.info("10fold model: {} done. ".format(name))
                results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold)
                mean= results.mean().round(3)
                std = results.std()
                logger.info("Model {} 10fold: mean:{},\t std:{}.".format(name, mean, std))
                myDF = myDF._append({'classifier': name, f'ten_f_CV{ij+1}':mean}, ignore_index = True)
            Tf_CV = pd.concat([Tf_CV, myDF], axis = 1)

            myDF1 = pd.DataFrame(columns=[])
            myDF2 = pd.DataFrame(columns=[])
            myDF3 = pd.DataFrame(columns=[])
            myDF4 = pd.DataFrame(columns=[])
            for name, model in tqdm(models, desc="myDF1"):   
                logger.info("Start training {}...".format(name))  
                model = model.fit(X_train, Y_train)
                logger.info("Train model: {} done. Then start predicting X_train...".format(name))
                Y_train_pred = model.predict(X_train)
                Y_test_pred = model.predict(X_test)
                logger.info("{} predict done.".format(name))
                Tr_precision = precision_score(Y_train, Y_train_pred, average="macro").round(3)
                Tr_recall = recall_score(Y_train, Y_train_pred, average="macro").round(3)
                Tr_f1 = f1_score (Y_train, Y_train_pred, average="macro").round(3)
                logger.info("Model {} X_train: Tr_precision:{},\t Tr_recall:{},\t Tr_f1:{}.".format(name, Tr_precision, Tr_recall, Tr_f1))
                myDF1 = myDF1._append({'classifier': name, f'tr_precision{ij+1}': Tr_precision, f'tr_recall{ij+1}': Tr_recall, f'tr_f1 {ij+1}':Tr_f1}, ignore_index = True)
                logger.info("{} predict done.".format(name))
                # report = sklearn.metrics.classification_report(Y_test, Y_test_pred)
                Te_precision = precision_score(Y_test, Y_test_pred, average='macro').round(3)
                Te_recall = recall_score(Y_test, Y_test_pred, average='macro').round(3)
                Te_f1 = f1_score (Y_test, Y_test_pred, average='macro').round(3)
                logger.info("Model {} X_train: Te_precision:{},\t Te_recall:{},\t Te_f1:{}.".format(name, Te_precision, Te_recall, Te_f1))
                myDF2 = myDF2._append({'classifier': name, f'te_precision{ij+1}': Te_precision, f'te_recall{ij+1}': Te_recall, f'te_f1 {ij+1}':Te_f1}, ignore_index = True)
                y_pred_proba3 = model.predict_proba(X_test)[::,1]
                logger.info("y_pred_proba: {} done. ".format(name))
                # keep probabilities for the positive outcome only
                fpr, tpr, _ = roc_curve(Y_test,  y_pred_proba3, pos_label = None)
                logger.info("Model {} AU_ROC: fpr:{},\t tpr:{}.".format(name, fpr, tpr))
                a_u_c = roc_auc_score(Y_test, y_pred_proba3).round(3)
                myDF3 = myDF3._append({'classifier': name, f'au ROC {ij+1}': a_u_c}, ignore_index = True)
                logger.info("Start predict probabilities {}...".format(name))
                y_pred_proba4 = model.predict_proba(X_test)
                # keep probabilities for the positive outcome only
                logger.info("y_pred_proba: {} done. ".format(name))
                y_pred_proba4 = y_pred_proba4[:, 1]
                #predict class vlaues
                y_pred4 = model.predict(X_test)
                # calculate precision-recall curve
                logger.info("Predict: {} done. calculate precision-recall curve...".format(name))
                precision, recall, _ = precision_recall_curve(Y_test, y_pred_proba4)
                # calculate au precision-recall curve
                logger.info("Predict: {} done. calculate au precision-recall curve...".format(name))
                area = auc(recall, precision).round(3)
                # calculate f1 score
                logger.info("Predict: {} done. calculate f1 score...".format(name))
                f1 = f1_score(Y_test, y_pred4, average='weighted').round(3)
                logger.info("Model {} AU_ROC: precision:{},\t recall:{},\t area:{},\t f1:{}.".format(name, precision, recall,area,f1))
                myDF4 = myDF4._append({'classifier': name, f'au PR {ij+1}': area}, ignore_index = True)

            Training_Performance = pd.concat([Training_Performance, myDF1], axis = 1)
            Test_Performance = pd.concat([Test_Performance, myDF2], axis = 1)
            Area_Under_ROC = pd.concat([Area_Under_ROC, myDF3], axis = 1)
            Area_Under_Precision_Recall = pd.concat([Area_Under_Precision_Recall, myDF4], axis = 1)
            ij += 1
                #List for output
        myDF5 = pd.DataFrame(columns=[])
        Loo = []

        #Leave one out validation
        cv = LeaveOneOut()

        #Evaluate each model
        for name, model in models:
            # fit model
            logger.info("Evaluating model {}...".format(name))
            # print("Evaluating model {}...".format(name))
            scores = cross_val_score(model, X, Y, cv = cv, scoring='accuracy', n_jobs=8).mean().round(3)
            myDF5 = myDF5._append({'classifier': name, f'Loo_CV {i+1}': scores}, ignore_index = True)
        Loo_CV = pd.concat([Loo_CV, myDF5], axis=1)
        Loo_CV.to_csv(f'{bacteria}_{antb}_Random_Set_Test_Loo_CV_{validation_no}-fold_CV.csv', encoding='utf-8')  


        Tf_CV.to_csv(f'{bacteria}_{antb}_Random_Set_Performance_Tf_CV_{validation_no}-fold_CV.csv', encoding='utf-8') 
        Training_Performance.to_csv(f'{bacteria}_{antb}_Random_Set_Performance_Training_Performance_{validation_no}-fold_CV.csv', encoding='utf-8') 
        Test_Performance.to_csv(f'{bacteria}_{antb}_Random_Set_Test_Training_Performance_{validation_no}-fold_CV.csv', encoding='utf-8') 
        Area_Under_ROC.to_csv(f'{bacteria}_{antb}_Random_Set_Test_Area_Under_ROC_{validation_no}-fold_CV.csv', encoding='utf-8') 
        Area_Under_Precision_Recall.to_csv(f'{bacteria}_{antb}_Random_Set_Test_Area_Under_Precision_Recall_{validation_no}-fold_CV.csv', encoding='utf-8')
    

  
        #Model names
        Models = Tf_CV.iloc[:, 0]
        logger.info("Calculating Performance")
        #Training_Performance F1 average
        tr_f1_avg = Training_Performance.filter(like='tr_f1').mean(axis=1).round(3)
        tr_f1_avg = tr_f1_avg.rename('tr_f1_avg', inplace=True)

        #Training_Performance precision average
        tr_precision_avg = Training_Performance.filter(like='tr_precision').mean(axis=1).round(3)
        tr_precision_avg = tr_precision_avg.rename('tr_precision_avg', inplace=True)

        #Training_Performance recall average
        tr_recall_avg = Training_Performance.filter(like='tr_recall').mean(axis=1).round(3)
        tr_recall_avg = tr_recall_avg.rename('tr_recall_avg', inplace=True)

        #Test_Performance f1 average
        te_f1_avg = Test_Performance.filter(like='te_f1').mean(axis=1).round(3)
        te_f1_avg = te_f1_avg.rename('te_f1_avg', inplace=True)

        #Test_Performance precision average
        te_precision_avg =  Test_Performance.filter(like='te_precision').mean(axis=1).round(3)
        te_precision_avg = te_precision_avg.rename('te_precision_avg', inplace=True)

        #Test_Performance recall average
        te_recall_avg = Test_Performance.filter(like='te_recall').mean(axis=1).round(3)
        te_recall_avg = te_recall_avg.rename('te_recall_avg', inplace=True)

        #Ten fold crossvalidation average
        Tf_CV_Avg = Tf_CV.filter(like='ten_f_CV').mean(axis=1).round(3)
        Tf_CV_Avg = Tf_CV_Avg.rename('Tf_CV_Avg', inplace=True)

        #Loo crossvalidation average
        Loo_CV_Avg = Loo_CV.filter(like='Loo_CV').mean(axis=1).round(3)
        Loo_CV_Avg = Loo_CV_Avg.rename('Loo_CV_Avg', inplace=True)
        # print(Loo_CV_Avg)
        #Area_Under_ROC average
        au_ROC_avg = Area_Under_ROC.filter(like='au ROC').mean(axis=1).round(3)
        au_ROC_avg = au_ROC_avg.rename('au_ROC_avg', inplace=True)

        #Area_Under_Precision_Recall average
        au_PR_avg = Area_Under_Precision_Recall.filter(like='au PR').mean(axis=1).round(3)
        au_PR_avg = au_PR_avg.rename('au_PR_avg', inplace=True)

        #Concatenate results
        logger.info("Concatenate results")
        frames1 = [Models, tr_precision_avg, tr_recall_avg, tr_f1_avg, te_precision_avg, te_recall_avg, te_f1_avg, Tf_CV_Avg, au_ROC_avg, au_PR_avg]

        Ran_Resul = pd.concat(frames1, axis=1)

        Random_Set_Results = pd.concat([Random_Set_Results, Ran_Resul, Loo_CV], axis =1)

    

    #Calculating average for outer 10 random sets from nested inner fold validation 
    logger.info("Calculating average for outer 10 random sets from nested inner fold validation")
    Models = pd.DataFrame(Models)

        #Training_Performance precision average
    tr_pa = Random_Set_Results['tr_precision_avg'].mean(axis=1).round(3)
    tr_pa = tr_pa.rename('tr_precision_avg', inplace=True)
    tr_pa = pd.DataFrame(tr_pa)

    #Training_Performance recall average
    tr_ra = Random_Set_Results['tr_recall_avg'].mean(axis=1).round(3)
    tr_ra = tr_recall_avg.rename('tr_recall_avg', inplace=True)
    tr_ra = pd.DataFrame(tr_ra)

    #Training_Performance F1 average
    tr_fa = Random_Set_Results['tr_f1_avg'].mean(axis=1).round(3)
    tr_fa = tr_fa.rename('tr_f1_avg', inplace=True)
    tr_fa = pd.DataFrame(tr_fa)

    #Test_Performance precision average
    te_pa = Random_Set_Results['te_precision_avg'].mean(axis=1).round(3)
    te_pa = te_pa.rename('te_precision_avg', inplace=True)
    te_pa = pd.DataFrame(te_pa)

    #Test_Performance recall average
    te_ra = Random_Set_Results['te_recall_avg'].mean(axis=1).round(3)
    te_ra = te_ra.rename('te_recall_avg', inplace=True)
    tr_ra = pd.DataFrame(tr_ra)

    #Test_Performance f1 average
    te_fa = Random_Set_Results['te_f1_avg'].mean(axis=1).round(3)
    te_fa = te_fa.rename('te_f1_avg', inplace=True)
    te_fa = pd.DataFrame(te_fa)

    #Ten fold crossvalidation average
    Tf_Ca = Random_Set_Results['Tf_CV_Avg'].mean(axis=1).round(3)
    Tf_Ca = Tf_Ca.rename('Tf_CV_Avg', inplace=True)
    Tf_Ca = pd.DataFrame(Tf_Ca)

    #Leave one out (Loo) crossvalidation average
    # Loo_Ca = Random_Set_Results.iloc[:,[191, 193, 195,197,199,201,203,205,207,209]].mean(axis=1).round(3)
    Loo_Ca = Random_Set_Results.filter(like='Loo_CV').mean(axis=1).round(3)
    Loo_Ca = Loo_Ca.rename('Loo_CV_Avg', inplace=True)
    Loo_Ca = pd.DataFrame(Loo_Ca)

    #Area_Under_ROC average
    au_Ra = Random_Set_Results['au_ROC_avg'].mean(axis=1).round(3)
    au_Ra = au_Ra.rename('au_ROC_avg', inplace=True)
    au_Ra = pd.DataFrame(au_Ra)

    #Area_Under_Precision_Recall average
    au_Pa = Random_Set_Results['au_PR_avg'].mean(axis=1).round(3)
    au_Pa = au_Pa.rename('au_PR_avg', inplace=True)
    au_Pa = pd.DataFrame(au_Pa)
    janakDF = pd.DataFrame(au_Pa)

    #Concatenate results
    logger.info("Concatenate Random Set Results")
    Random_Set_Results = pd.concat([ Models, tr_pa, tr_ra, tr_fa, te_pa, te_ra, te_fa, Tf_Ca, Loo_Ca, au_Ra, au_Pa], axis=1)

    # print(f'Random Set Results {antb} {bacteria}')
    # display(Random_Set_Results)

    #Result out
    logger.info("Save Random Set Results")
    with open (file_all_out, 'a+') as rf:
        rf.write(f'\n{bacteria} {antb} Random set Results\n')
        Random_Set_Results.to_csv(rf)

    #Export result separately
    logger.info("Export result separately")
    Random_Set_Results.to_csv(f'{bacteria}_{antb}_Random_Set_Performance_{validation_no}-fold_CV.csv', encoding='utf-8')
    
    #Plot All set, Intersection set, Random set performance comparision figures
    logger.info("Plot All set, Intersection set, Random set performance comparision figures")

    #Open files to dataframe
    d1 = pd.read_csv(f'{bacteria}_{antb}_All_Set_Performance_{validation_no}-fold_CV.csv')
    d2 = pd.read_csv(f'{bacteria}_{antb}_Intersection_Set_Performance_{validation_no}-fold_CV.csv' )
    d3 = pd.read_csv(f'{bacteria}_{antb}_Random_Set_Performance_{validation_no}-fold_CV.csv')

    #Select classifier names
    models = d1[['classifier']]

    #Training precision
    a_s = d1[['tr_precision_avg']]
    i_s = d2[['tr_precision_avg']]
    r_s = d3[['tr_precision_avg']]
    df1 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df1.set_index(['classifier'], inplace=True)

    #Training recall
    a_s = d1[['tr_recall_avg']]
    i_s = d2[['tr_recall_avg']]
    r_s = d3[['tr_recall_avg']]
    df2 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df2.set_index(['classifier'], inplace=True)

    #Training f1
    a_s = d1[['tr_f1_avg']]
    i_s = d2[['tr_f1_avg']]
    r_s = d3[['tr_f1_avg']]
    df3 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df3.set_index(['classifier'], inplace=True)

    #Test precision
    a_s = d1[['te_precision_avg']]
    i_s = d2[['te_precision_avg']]
    r_s = d3[['te_precision_avg']]
    df4 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df4.set_index(['classifier'], inplace=True)

    #Test recall
    a_s = d1[['te_recall_avg']]
    i_s = d2[['te_recall_avg']]
    r_s = d3[['te_recall_avg']]
    df5 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df5.set_index(['classifier'], inplace=True)

    #Test f1
    a_s = d1[['te_f1_avg']]
    i_s = d2[['te_f1_avg']]
    r_s = d3[['te_f1_avg']]
    df6 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df6.set_index(['classifier'], inplace=True)

    #Export separate dataframe of f1
    df6.to_csv(f'{bacteria}_{antb}_F1_comparision_{validation_no}-fold_CV.csv', encoding='utf-8')

    #10f CV
    a_s = d1[['Tf_CV_Avg']]
    i_s = d2[['Tf_CV_Avg']]
    r_s = d3[['Tf_CV_Avg']]
    df7 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df7.set_index(['classifier'], inplace=True)

    #Loo_CV
    a_s = d1[['Loo_CV']]
    i_s = d2[['Loo_CV']]
    r_s = d3[['Loo_CV_Avg']].rename(columns={"Loo_CV_Avg": "Loo_CV"})
    df8 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df8.set_index(['classifier'], inplace=True)

    #Au_ROC
    a_s = d1[['au_ROC_avg']]
    i_s = d2[['au_ROC_avg']]
    r_s = d3[['au_ROC_avg']]
    df9 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df9.set_index(['classifier'], inplace=True)

    #Au_PR
    a_s = d1[['au_PR_avg']]
    i_s = d2[['au_PR_avg']]
    r_s = d3[['au_PR_avg']]
    df10 = pd.concat([models, a_s, i_s, r_s], axis =1)
    df10.set_index(['classifier'], inplace=True)

    #Bar diagram colors and labels
    my_colors=['#13203c', '#fca412', '#d0d0d0']
    my_labels=['All Set', 'Intersection Set', 'Random Set']

    #Activate latex text rendering
    rc('text', usetex=False)
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=False, sharey=False, figsize=(15,10))

    plt.xlabel("")
    plt.margins(y=0)

    ax1 = df1.plot(kind='bar', ax=axes[0,0], color=my_colors, width=0.7, edgecolor='grey', 
             linewidth=0.5, title= "i. Training precisiion",
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax1.set_xlabel('')
    ax1.grid(linestyle='-', color='#666666', axis='y', alpha=0.3, lw=0.5)
    ax1.set_axisbelow(True)
    ax1.margins(0)

    ax2 = df2.plot(kind='bar', ax=axes[0,1], color=my_colors, width=0.7, edgecolor='grey', 
             linewidth=0.5, title ="ii. Training recall", 
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax2.set_xlabel('')
    ax2.margins(0)
    ax2.grid(linestyle='-', color='#666666', axis='y', alpha=0.3, lw=0.5)
    ax2.set_axisbelow(True)

    ax3 = df3.plot(kind='bar', ax=axes[0,2], color=my_colors, width=0.7, edgecolor='grey', 
             linewidth=0.5, title="iii. Training f1", 
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax3.set_xlabel('')
    ax3.margins(0)
    ax3.grid(linestyle='-', color ='#666666', axis='y', alpha=0.3, lw=0.5)
    ax3.set_axisbelow(True)

    ax4 = df4.plot(kind='bar', ax=axes[1,0], color=my_colors, width=0.7, edgecolor='grey', 
             linewidth=0.5, title ="iv. Test precision", 
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax4.set_xlabel('')
    ax4.grid(linestyle='-', color ='#666666', axis='y', alpha=0.3, lw=0.5)
    ax4.set_axisbelow(True)

    ax5 = df5.plot(kind='bar', ax=axes[1,1], color=my_colors, width=0.7, edgecolor='grey', 
             linewidth=0.5, title ="v. Test recall", 
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax5.set_xlabel('')
    ax5.grid(linestyle='-', color ='#666666', axis='y', alpha=0.3, lw=0.5)
    ax5.set_axisbelow(True)

    ax6 = df7.plot(kind='bar', ax=axes[1,2], color=my_colors, width = 0.7, edgecolor='grey', 
             linewidth=0.5, title="vi. 10-fold CV", 
             legend=False, yticks=np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax6.set_xlabel('')
    ax6.grid(linestyle='-', color ='#666666', axis='y', alpha=0.3, lw=0.5)
    ax6.set_axisbelow(True)

    ax7 = df8.plot(kind='bar', ax=axes[2,0], color=my_colors, width=0.7, edgecolor='grey', 
             linewidth=0.5, title="vii. Loo CV", 
             legend=False, yticks = np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax7.set_xlabel('')
    ax7.grid(linestyle='-', color ='#666666', axis='y', alpha=0.3, lw=0.5)
    ax7.set_axisbelow(True)

    ax8 = df9.plot(kind='bar', ax=axes[2,1], color=my_colors, width=0.7, edgecolor='grey', 
             linewidth=0.5, title ="viii. au ROC", 
             legend=False, yticks = np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax8.set_xlabel('')

    ax8.legend(labels=my_labels, loc='lower center', borderaxespad=0.1, ncol=3,
               bbox_to_anchor=(0.5, -0.25),
               fancybox=False, shadow=False, prop={'size': 8})
    ax8.grid(linestyle='-', color ='#666666', axis='y', alpha=0.3, lw=0.5)
    ax8.set_axisbelow(True)

    ax9 = df10.plot(kind='bar', ax=axes[2,2], color=my_colors, width=0.7, edgecolor='grey', 
             linewidth=0.5, title ="i. au PR", 
             legend=False, yticks = np.arange(0,1.1, step=0.1),
             fontsize=8, rot=0)
    ax9.set_xlabel('')
    ax9.grid(linestyle='-', color='#666666', axis='y', alpha=0.3, lw=0.5)
    ax9.set_axisbelow(True)

    t = f"Suplementary Figure {supplementary_fig_no}. Assessment of the performance of the machine learning algorithms in predicting resistance to {antb} by {italic_name} in {validation_no}-fold cross validation settings. The preformance metrics i) training precision, \nii) training recall, iii) training F1, iv) test precision, v) test recall, vi) 10f CV (ten-fold cross validation), vii) Loo CV (leave-one-out cross validation), viii) au ROC (area under ROC curve) , and ix) au PR (area under precision recall curve). \n'All'denotes all AMR genes for taraining (as in the cross-validation partioning), 'Intersection' refers to AMR genes that consistently ranked high across all 6 rounds of cross-validation, and 'Random' refers to randomly sampled AMR genes."

    import textwrap as tw
    fig_txt= tw.fill(tw.dedent(t.strip() ), width=250)

    plt.figtext(0.5, 0.03, fig_txt, horizontalalignment='center',
                fontsize=10, multialignment='left',
                bbox=dict(boxstyle="round", facecolor='lavender', lw=0.5, pad=0.5, alpha=0.5, 
                          edgecolor='grey', linewidth=0.5))
    fig.tight_layout()

    plt.subplots_adjust(top=0.97, bottom=0.15, hspace=0.29, wspace=0.1 )
    fig.savefig(f'ML_Plot_{bacteria}_{antb}_{validation_no}-fold_CV.tiff', dpi=300, format="tiff", pil_kwargs={"compression": "tiff_lzw"})

    plt.show()

#Dictionary of antibiotics
antb_SN = {'rifampicin': '1 a'}

for antb, SN in antb_SN.items():
    
    #Acronym for Klebsiella pneumoniae
    bacteria = 'MTB'
    
    #Italicized full name bacteria for fig output
    italic_name = r"\textit{Mycobacterium tuberculosis}"
    
    #Figure plot number
    supplementary_fig_no = SN
    
    #Import amr-ast data from github repository
    file_name = f'./Data/{antb}.csv'

    #Import bacterial strains without caapenemase from github repository
    To_predict = f'./Data/{antb}_to_predict_MTB.csv'
    
    #no of validation
    validation_no = 6

    # loggingfile config
    logger = logging.getLogger(antb)
    logger.setLevel(logging.INFO)
    rf_handler = logging.StreamHandler(sys.stderr)
    rf_handler.setLevel(logging.DEBUG) 
    #rf_handler = logging.handlers.TimedRotatingFileHandler('all.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
    rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(message)s"))
    f_handler = logging.FileHandler('{}_{}.txt'.format(antb, time.strftime("%Y-%m-%d%H:%M:%S", time.localtime())))
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
    
    logger.addHandler(rf_handler)
    logger.addHandler(f_handler)
        
    ML_run()