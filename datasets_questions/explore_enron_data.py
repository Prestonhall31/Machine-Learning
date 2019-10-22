#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import sys
import pickle
sys.path.append("../tools")
# from feature_format import featureFormat
# from feature_format import targetFeatureSplit

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))


def poi_count(data):
    """ 
    Returns an integer

    This method takes a dictionary or dictionaries dataset and iterates through 
    to see which data points are persons of interest (poi).
    """
    poi = 0

    for k, v in data.items():
        if v['poi']:
            poi += 1
    
    return poi


def find_salary_and_email(data):
    salary = 0
    email = 0

    for k, v in data.items():
        if v['salary'] != "NaN":
            salary += 1

    for k, v in data.items():
        if v['email_address'] != "NaN":
            email += 1
        
    return salary, email


def tp_NaN(data):
    tot_pay = 0

    for k, v in data.items():
        if v['total_payments'] == "NaN":
            tot_pay += 1

    return tot_pay


# print(enron_data.keys())

# Find the amount of POI
print("There are", poi_count(enron_data), "persons of interest.")

# What is the stock value for James Prentice
print("James Prentice has a stock value of", '${:,.0f}'.format(enron_data['PRENTICE JAMES']['total_stock_value']))

# How many email messages do we have from Wesley Colwell to persons of interest?
print("Wesley Colwell sent", enron_data['COLWELL WESLEY']['from_this_person_to_poi'], "emails to persons of interest.")

# What’s the value of stock options exercised by Jeffrey K Skilling?
print("Jeffrey K Skilling excersised stock value: ", 
    '${:,.0f}'.format(enron_data['SKILLING JEFFREY K']['exercised_stock_options']))

# Who took home the most money?
print("Andrew Fastow:", '${:,.0f}'.format(enron_data['FASTOW ANDREW S']['total_payments']), "\n",
"Jeffrey K Skilling:", '${:,.0f}'.format(enron_data['SKILLING JEFFREY K']['total_payments']), "\n",
"Kenneth Lay:", '${:,.0f}'.format(enron_data['LAY KENNETH L']['total_payments']), "\n")

#How many folks in this dataset have a quantified salary? What about a known email address?
print(find_salary_and_email(enron_data))

'''
We’ve written some helper functions (featureFormat() and targetFeatureSplit() in 
tools/feature_format.py) that can take a list of feature names and the data dictionary, 
and return a numpy array.

In the case when a feature does not have a value for a particular person, this function 
will also replace the feature value with 0 (zero).
'''


###  OPTIONAL ###

# 1. How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments? 
# What percentage of people in the dataset as a whole is this?
print("Total NaN:", tp_NaN(enron_data))
print("Percentage of total:", tp_NaN(enron_data)/len(enron_data.keys()), "\n")

# 2. How many POIs in the E+F dataset have “NaN” for their total payments? What percentage of POI’s 
# as a whole is this?

# 3. If a machine learning algorithm were to use total_payments as a feature, 
# would you expect it to associate a “NaN” value with POIs or non-POIs?

# 4. If you added in, say, 10 more data points which were all POI’s, and put “NaN” for the 
# total payments for those folks, the numbers you just calculated would change.
# What is the new number of people of the dataset? What is the new number of folks with “NaN” for total payments?

# 5. What is the new number of POI’s in the dataset? What is the new number of POI’s with NaN for total_payments?

# 6. Once the new data points are added, do you think a supervised classification algorithm might interpret 
# “NaN” for total_payments as a clue that someone is a POI?

'''
Adding in the new POI’s in this example, none of whom we have financial information for, has introduced a 
subtle problem, that our lack of financial information about them can be picked up by an algorithm as a 
clue that they’re POIs. Another way to think about this is that there’s now a difference in how we generated 
the data for our two classes--non-POIs all come from the financial spreadsheet, while many POIs get added in 
by hand afterwards. That difference can trick us into thinking we have better performance than we do--suppose 
you use your POI detector to decide whether a new, unseen person is a POI, and that person isn’t on the 
spreadsheet. Then all their financial data would contain “NaN” but the person is very likely not a POI 
(there are many more non-POIs than POIs in the world, and even at Enron)--you’d be likely to accidentally 
identify them as a POI, though!

This goes to say that, when generating or augmenting a dataset, you should be exceptionally careful 
if your data are coming from different sources for different classes. It can easily lead to the type 
of bias or mistake that we showed here. There are ways to deal with this, for example, you wouldn’t 
have to worry about this problem if you used only email data--in that case, discrepancies in the 
financial data wouldn’t matter because financial features aren’t being used. There are also more 
sophisticated ways of estimating how much of an effect these biases can have on your final answer; 
those are beyond the scope of this course.

For now, the takeaway message is to be very careful about introducing features that come from different 
sources depending on the class! It’s a classic way to accidentally introduce biases and mistakes.

'''

