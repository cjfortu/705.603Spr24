import pandas as pd
import holidays
import pickle


class Point_Pipeline:
    """
    A class to normalize, stem, tokenize, encode, and load data for machine learning.
    """
    def __init__(self):
        self.us_hols = holidays.US(years = range(2015, 2050))
        
    
    def loaddat(self):
        """
        Load the mapping from usertypes to Customer ID's
        along with the processed success and failure emails.
        """
        with open ('./data/ubseq.bin', 'rb') as f:
            ubseq = pickle.load(f)
        with open ('./data/dfres.bin', 'rb') as f:
            dfres = pickle.load(f)
        
        return ubseq, dfres
        
        
    def proc_email(self, email):
        """
        A helper for binary encoding the email domain.
        
        parameters:
        email (str): the email address
        
        returns:
        int: The bit to flip, if any.
        """
        email = email.lower()
        domain = email.split('@')[1].split('.')[0]
        if domain == 'aol':
            return 2
        if domain == 'comcast':
            return 3
        if domain == 'gmail':
            return 4
        if domain == 'hotmail':
            return 5
        if domain == 'msn':
            return 6
        else:
            return None
        
        
    def proc_age(self, age):
        """
        A helper for binary encoding age category.
        
        parameters:
        age (int): the customer age
        
        returns:
        int: The bit to flip, if any.
        """
        if age == 18:
            return 7
        elif age >= 19 and age <= 22:
            return 8 
        elif age >= 23 and age <= 29:
            return 9
        elif age >= 30 and age <= 39:
            return 10
        elif age >= 40 and age <= 49:
            return 11
        elif age >= 50 and age <= 59:
            return 12
        else:
            return None
        
        
    def proc_ten(self, tenure):
        """
        A helper for binary encoding the tenure category
        
        parameters:
        tenure (int): the tenure
        
        returns:
        int: The bit to flip, if any.
        """
        if tenure <= 3:
            return 13
        elif tenure >= 4 and tenure <= 9:
            return 14
        elif tenure >= 10 and tenure <= 15:
            return 15
        elif tenure >= 16 and tenure <= 23:
            return 16
        elif tenure >= 24 and tenure <= 26:
            return 17
        else:
            return None
        
        
    def proc_usr(self, usrfeats):
        """
        Get a user type index based on user features
        
        parameters:
        usrfeats (list of str): The user information
        
        returns:
        utypidx (int): The user index based on binary existence of features
        """
        gender, age, custtype, email, tenure = usrfeats
        custbin = list('000000000000000000')  # the 18 customer features
        # flip gender and type bits as necessary
        if gender == 'F':
            custbin[0] = '1'
        if custtype == 'B':
            custbin[1] = '1'
           
        domidx = self.proc_email(email)
        if domidx != None:  # flip email domain bits as necessary
            custbin[domidx] = '1'
        
        agidx = self.proc_age(int(age))
        if agidx != None: # flip age bits as necessary
            custbin[agidx] = '1'
            
        tenidx = self.proc_ten(int(tenure))
        if tenidx != None:  # flip tenure bits as necessary
            custbin[tenidx] = '1'
    
        # convert binary string back to integer
        binstr = ''.join(custbin)
        utypidx = int(binstr, 2)
        
        return utypidx

        
    def proc_date(self, date):
        """
        Convert the date into a day type index
        
        parameters:
        date (str): YYYY-MM-DD
        
        returns:
        wdfrihol (int): The index of the day type
        """
        dateobj = pd.to_datetime(date)
        
        # the following are bits representing the day types
        wknd = str(int(dateobj.dayofweek > 4))
        fri = str(int(dateobj.dayofweek == 4))
        hol = str(int(dateobj in self.us_hols))
        
        # now encode the binary into an integer representation
        wdfriholbin = ''.join([hol, fri, wknd])
        wdfrihol = int(wdfriholbin, 2)
        
        return wdfrihol
    