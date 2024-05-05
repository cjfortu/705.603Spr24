

def get_utypes(dfub, icb):
    matches = list(dfub.loc[
        (dfub['Gender_F']==icb[0]) & (dfub['Type_B']==icb[1]) &
        (dfub['Domain_aol']==icb[2]) & (dfub['Domain_comcast']==icb[3]) &
        (dfub['Domain_gmail']==icb[4]) & (dfub['Domain_hotmail']==icb[5]) &
        (dfub['Domain_msn']==icb[6]) & (dfub['AgeG_18']==icb[7]) &
        (dfub['AgeG_19-22']==icb[8]) & (dfub['AgeG_23-29']==icb[9]) &
        (dfub['AgeG_30-39']==icb[10]) & (dfub['AgeG_40-49']==icb[11]) &
        (dfub['AgeG_50-59']==icb[12]) & (dfub['TenG_0-3']==icb[13]) &
        (dfub['TenG_4-9']==icb[14]) & (dfub['TenG_10-15']==icb[15]) &
        (dfub['TenG_16-23']==icb[16]) & (dfub['TenG_24-26']==icb[17]),
        'Customer_ID'])
    
    ubseqkey = '{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}'.format(icb[0], icb[1], icb[2], icb[3],
                                                 icb[4], icb[5], icb[6], icb[7],
                                                 icb[8], icb[9], icb[10], icb[11],
                                                 icb[12], icb[13], icb[14], icb[15], icb[16], icb[17])
    with open('output.txt', 'a') as f:
        f.write(ubseqkey)
    
    return ubseqkey, matches
    
 