import pandas as pd
import matplotlib.pyplot as plt
from pandasql import sqldf
import numpy as np


if __name__ == '__main__':
    data = pd.read_csv("dataset_mood_smartphone.csv")
    ids = sqldf("SELECT DISTINCT id FROM data")
    ids = ids["id"].astype(str).tolist()
    patients = []
    for id in ids:
        patients.append(sqldf("SELECT * FROM data "
               "WHERE id IS '" + id +"'"))
    #print(patients)

    moods = []
    circumplex_arousals = []
    circumplex_valences = []
    activitys = []
    screens = []
    calls = []
    smss = []
    builtins = []
    communications = []
    entertainments = []
    finances = []
    games = []
    offices = []
    others = []
    socials = []
    travels = []
    unknowns = []
    utilitiess = []
    weathers = []

    for patient in patients:
        moods.append(sqldf("SELECT variable as mood, value, time FROM patient "
          "WHERE variable IS 'mood'"))
        circumplex_arousals.append(sqldf("SELECT variable as 'circumplex.arousal', value, time FROM patient "
          "WHERE variable IS 'circumplex.arousal'"))
        circumplex_valences.append(sqldf("SELECT variable as 'circumplex.valence', value, time FROM patient "
          "WHERE variable IS 'circumplex.valence'"))
        activitys.append(sqldf("SELECT variable as activity, value, time FROM patient "
          "WHERE variable IS 'activity'"))
        screens.append(sqldf("SELECT variable as screen, value, time FROM patient "
          "WHERE variable IS 'screen'"))
        calls.append(sqldf("SELECT variable as call, value, time FROM patient "
          "WHERE variable IS 'call'"))
        smss.append(sqldf("SELECT variable as sms, value, time FROM patient "
          "WHERE variable IS 'sms'"))
        builtins.append(sqldf("SELECT variable as 'appCat.builtin', value, time FROM patient "
            "WHERE variable IS 'appCat.builtin'"))
        communications.append(sqldf("SELECT variable as 'appCat.communication', value, time FROM patient "
          "WHERE variable IS 'appCat.communication'"))
        entertainments.append(sqldf("SELECT variable as 'appCat.entertainment', value, time FROM patient "
          "WHERE variable IS 'appCat.entertainment'"))
        finances.append(sqldf("SELECT variable as 'appCat.finance', value, time FROM patient "
          "WHERE variable IS 'appCat.finance'"))
        games.append(sqldf("SELECT variable as 'appCat.game', value, time FROM patient "
          "WHERE variable IS 'appCat.game'"))
        offices.append(sqldf("SELECT variable as 'appCat.office', value, time FROM patient "
          "WHERE variable IS 'appCat.office'"))
        others.append(sqldf("SELECT variable as 'appCat.other', value, time FROM patient "
          "WHERE variable IS 'appCat.other'"))
        socials.append(sqldf("SELECT variable as 'appCat.social', value, time FROM patient "
          "WHERE variable IS 'appCat.social'"))
        travels.append(sqldf("SELECT variable as 'appCat.travel', value, time FROM patient "
          "WHERE variable IS 'appCat.travel'"))
        unknowns.append(sqldf("SELECT variable as 'appCat.unknown', value, time FROM patient "
          "WHERE variable IS 'appCat.unknown'"))
        utilitiess.append(sqldf("SELECT variable as 'appCat.utilities', value, time FROM patient "
          "WHERE variable IS 'appCat.utilities'"))
        weathers.append(sqldf("SELECT variable as 'appCat.weather', value, time FROM patient "
          "WHERE variable IS 'appCat.weather'"))

    p0_mood = sqldf("SELECT variable as mood, value, time FROM p0 "
          "WHERE variable IS 'mood'")
    p0_activity = sqldf("SELECT variable as activity, value, time FROM p0 "
          "WHERE variable IS 'activity'")

    p0_avg_mood = sqldf("SELECT AVG(value) FROM p0_mood "
      "WHERE time BETWEEN '2014-04-19 00:00:00.000' and '2014-04-22 00:00:00.000'")

    print(p0_avg_mood)

