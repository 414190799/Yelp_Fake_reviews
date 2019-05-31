# coding: utf-8
# !/usr/bin/python3
"""
Authors: Jiajun Bao, Meng Li, Jane Liu
Classes:
    MainWindow: The detecting system window. Take the review and rating as input and then give the
        predict value and predict label. Please note that this deployment program does not work with
        batch processing from a remote location (where you need to SSH to the server). The server
        will display an error for "Display device not found". This deployment program will work
        correctly with a desktop or laptop with sufficient memory to run the entire program. More
        details about this limitation is available at https://github.com/Kamnitsask/deepmedic/issues/1
"""


from tkinter import *
from tabulate import tabulate   # need to import
from nltk.stem import WordNetLemmatizer 
import math
import requests                 # need to import

#main window
class MainWindow:
    #get result function
        
    def getresult(self):   
        
        validate1=self._val.dropna()
        validate1=validate1[validate1['content']!="nan"]
        
        #generate topic features:
        
        
        for topic in self._ftopic:
            tlist=[]
            for c in validate1["content"]:
                count=0
                p=c.split()
                for word in p:
                    if word==topic:
                        count+=1
                tlist.append(count)
            validate1[topic]=tlist
        
        #create new row for the input data
        validate1.loc['new_row']=0
        
        validate1.loc["new_row","content"]=str(self.entry_rev.get())
        validate1.loc["new_row","rating"]=float(self.entry_rate.get())
        
        lemmatizer = WordNetLemmatizer()
        
        for topic in self._ftopic:
            count=0
            p=str(self.entry_rev.get()).split()
            for word in p:
                word=lemmatizer.lemmatize(word.lower())
                if word==topic:
                    count+=1
            validate1.loc["new_row",topic]=count
        
        #generate feature: "length of review":
        for elem in validate1.content:
            self._len.append(len(str(elem)))
        validate1["length_of_review"]=self._len
        
        #treat the feature "rating" as dummy variable
        dummy_ranks = pd.get_dummies(validate1['rating'], prefix='rating')
        
        #create dataset for regression
        
        self._cols_to_keep=["length_of_review"]
        
        for elem in self._ftopic:
            self._cols_to_keep.append(elem)
        
        val = validate1[self._cols_to_keep].join(dummy_ranks.ix[:, 'rating_2.0':])
             
        
        #standarized train data
        val=val.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
        
        #add intercept
        val['intercept'] = 1.0
        
        #run the model
        
        val['predict'] = self._result.predict(val)
        
                
        # if the value of predict is bigger than 0.5, assign it as non-fake, otherwise is fake
        val.loc[val.predict>0.5,'plabel'] ="truthful"
        val.loc[val.predict<=0.5,'plabel'] ="fake"           
            
 
        #visualization
        table1 = tabulate(val.tail(1).iloc[:,:-3],headers='keys')
        table2 = tabulate(val.tail(1).iloc[:,-2:],headers='keys')
        self.text.insert('end', table1)  
        self.text.insert('end', table2)  

    
    # GUI window commands
    def __init__(self,file,ftopic,result):
        self._val = file
        self._ftopic= ftopic
        self._result=result
        self._len = []
        self._cols_to_keep=[]

        
        self.frame = Tk()
        
        #window title
        self.frame.title('Detecting System')
        
        #window geometry
        self.frame.geometry('1300x600')
        
        #add labels 
        self.label_rev = Label(self.frame,text = "Reviews: ")
        self.label_rate = Label(self.frame,text = "Rating: ")
        
        #add entry part
        self.entry_rev = Entry(self.frame)
        self.entry_rate = Entry(self.frame)
       
        #add text part
        self.text = Text(self.frame,width=200)
        
        #add button 
        self.button_detect = Button(self.frame,text = "Detect",width = 10,command= self.getresult )
        self.button_cancel = Button(self.frame,text = "Cancel",width = 10 )
        
        #grid them
        self.label_rev.grid(row = 0,column = 7)
        self.label_rate.grid(row = 1,column = 7)
        
        
        self.entry_rev.grid(row = 0,column = 8)
        self.entry_rate.grid(row = 1,column = 8)
        
        
        self.button_detect.grid(row = 3,column = 7)
        self.button_cancel.grid(row = 3,column = 8)
        
        self.text.grid(row=6,columnspan=24)

        
        self.frame= mainloop()
