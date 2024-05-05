
class Emailenv:
    def __init__(self, ubseq, dfres, rewthresh, mode='Train'):
        self.ubseq = ubseq  # the dict mapping user types to customer IDs
        self.dfres = dfres  # dataframe of failure and success emails
        self.rewthresh = rewthresh  # the value of each valid response
        self.offset = 1e-20  # a small offset to prevent division by zero
        
        self.utypidx = 0  # start with the first usertype
        self.maxbatsteps = 2**18
        if mode == 'Train':
            self.utyps = list(ubseq.keys())
            self.get_next_valid_state()
            

    def get_next_valid_state(self):
        """
        Gets the next state with associated data.
        
        Necessary since some states have no associated data
        """
        self.curusrs = self.ubseq[self.utyps[self.utypidx]]
        # continue cycling through usertypes until we have customer IDs
        while len(self.curusrs) == 0:
            self.utypidx += 1
            if self.isdone() == True:
                return
            self.curusrs = self.ubseq[self.utyps[self.utypidx]]
        # use the customer IDs to get the success and failure emails
        self.res = self.dfres.loc[self.dfres['Customer_ID'].\
                isin(self.curusrs)].\
                reset_index()
        # the integer code for day type
        self.wdfrihol = 0  # up to 7 for wknd=True, fri=True, hol=True. 0 means all False
        # the integer code for the state
        self.stateidx = self.utypidx + (self.wdfrihol * 2**18)
    
    
    def get_state(self):
        return self.stateidx


    def isdone(self):
        if self.utypidx == self.maxbatsteps - 1:
            return True
        else:
            return False


    def proc_reward_conv(self, act):
        """
        Computes the reward and conversion numbers.
        
        Reward is designed so a state-action pair (for action != 0) must have
        a conversion >= 1/rewthresh in order to learn that action as
        opposed to action=0
        
        parameters:
        act (int): The action. 0=no-send, 1,2,3=email-subject
        
        returns:
        reward (int): The reward for the state-action pair.
        nresp (int): The number of valid responses.
        nsent (int): The number of sent emails.
        """
        if act != 0:  # send an email with one of three subject lines
            # the day type in binary
            wdfriholbin = '{0:03b}'.format(self.wdfrihol)
            # assign the binary values to holiday, friday, and weekend day types
            hol, fri, wknd = int(wdfriholbin[0]), int(wdfriholbin[1]), int(wdfriholbin[2])
            nfails = len(self.res.loc[(self.res['SubjectLine_ID'] == act) &\
                    (self.res['weekend'] == wknd) & (self.res['friday'] == fri) &\
                    (self.res['holiday'] == hol) & (self.res['resp'] == False)])
            nresp = len(self.res.loc[(self.res['SubjectLine_ID'] == act) &\
                    (self.res['weekend'] == wknd) & (self.res['friday'] == fri) &\
                    (self.res['holiday'] == hol) & (self.res['resp'] == True)])
            nsent = nfails + nresp + self.offset
            # reward is negative if the state-action pair has insuffucient responses
            if nsent != self.offset:
                reward = (-1 * nfails) + (self.rewthresh * nresp)
            else:
                reward = -1
        elif act == 0:  # do not send an email
            reward = 0
            nresp = 0
            nsent = self.offset

        return reward, nresp, nsent


    def step(self, act):
        """
        Take a step
        
        parameters:
        act (int): The action. 0=no-send, 1,2,3=email-subject
        
        returns:
        self.stateidx (int): The state index. A composite of the user type and day type.
        nresp (int): The number of valid responses.
        nsent (int): The number of sent emails.
        self.done (bool): True for a complete episode
        """
        self.wdfrihol += 1  # each step cycles through weekend, friday, or holiday day tyoes
        if self.wdfrihol == 8:  # move to next user type after cycling though day types
            self.utypidx += 1
            self.get_next_valid_state()
        else:
            self.stateidx = self.utypidx + (self.wdfrihol * 2**18)
        reward, nresp, nsent = self.proc_reward_conv(act)
        self.done = self.isdone()

        return self.stateidx, reward, nresp, nsent, self.done


    def reset(self):
        self.utypidx = 0  # go back to the first usertype
        self.stepn = 0
        self.get_next_valid_state()

    
    def handle_single(self, act, utypidx, wdfrihol):
        """
        Get the predicted conversion for a state-action pair
        
        parameters:
        act (int): The action.  0=no-send, 1,2,3=email-subject
        utypidx (int): The usertype index.  A component of the state
        wdfrihol (int): The day type.  A component of the state
        
        return:
        conversion (float): The conversion rate
        """
        curusrs = self.ubseq[self.utyps[utypidx]]
        res = self.dfres.loc[self.dfres['Customer_ID'].isin(curusrs)].reset_index()
        
        # the day type in binary
        wdfriholbin = '{0:03b}'.format(wdfrihol)
        # assign the binary values to holiday, friday, and weekend day types
        hol, fri, wknd = int(wdfriholbin[0]), int(wdfriholbin[1]), int(wdfriholbin[2])
        nfails = len(res.loc[(res['SubjectLine_ID'] == act) &\
                (res['weekend'] == wknd) & (res['friday'] == fri) &\
                (res['holiday'] == hol) & (res['resp'] == False)])
        nresp = len(res.loc[(res['SubjectLine_ID'] == act) &\
                (res['weekend'] == wknd) & (res['friday'] == fri) &\
                (res['holiday'] == hol) & (res['resp'] == True)])
        nsent = nfails + nresp + self.offset
        
        conversion = nresp / nsent
        
        return conversion