#!/usr/bin/python
import numpy as np
import pandas as pd
from csv import reader
from csv import writer
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import argparse
import sys
import json
import ssl
import urllib.request
import random


def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--countries',
        action='store',
        dest='countries',
        help='Countries on CSV format. ' +
        'It must exact match the data names or you will get out of bonds error.',
        metavar='COUNTRY_CSV',
        type=str,
        default="")
    
    parser.add_argument(
        '--download-data',
        action='store_true',
        dest='download_data',
        help='Download fresh data and then run',
        default=False
    )

    parser.add_argument(
        '--start-date',
        required=False,
        action='store',
        dest='start_date',
        help='Start date on MM/DD/YY format ... I know ...' +
        'It defaults to first data available 1/22/20',
        metavar='START_DATE',
        type=str,
        default="1/22/20")

    parser.add_argument(
        '--xDaysBefore',
        required=False,
        action='store',
        dest='dataFiveBefore',
        help='Start date on MM/DD/YY format ... I know ...' +
        'It defaults to first data available 1/27/20',
        metavar='dataFiveBefore',
        type=str,
        default="1/22/20")

    parser.add_argument(
        '--prediction-days',
        required=False,
        dest='predict_range',
        help='Days to predict with the model. Defaults to 150',
        metavar='PREDICT_RANGE',
        type=int,
        default=150)

    parser.add_argument(
        '--S_0',
        required=False,
        dest='s_0',
        help='S_0. Defaults to 100000',
        metavar='S_0',
        type=int,
        default=50000000)

    parser.add_argument(
        '--A_0',
        required=False,
        dest='a_0',
        help='A_0. Defaults to 0',
        metavar='A_0',
        type=int,
        default=random.randrange(0, 100, 1)) # need revision

    parser.add_argument(
        '--Y_0',
        required=False,
        dest='y_0',
        help='Y_0. Defaults to 2',
        metavar='Y_0',
        type=int,
        default=2)

    parser.add_argument(
        '--R_0',
        required=False,
        dest='r_0',
        help='R_0. Defaults to 0',
        metavar='R_0',
        type=int,
        default=0)

    parser.add_argument(
        '--D_0',
        required=False,
        dest='d_0',
        help='D_0. Defaults to 0',
        metavar='D_0',
        type=int,
        default=0)

    args = parser.parse_args()

    country_list = []
    if args.countries != "":
        try:
            countries_raw = args.countries
            country_list = countries_raw.split(",")
        except Exception:
            sys.exit("QUIT: countries parameter is not on CSV format")
    else:
        sys.exit("QUIT: You must pass a country list on CSV format.")

    return (country_list, args.download_data, args.start_date, args.dataFiveBefore, args.predict_range, args.s_0, args.a_0, args.y_0, args.r_0, args.d_0)


def sumCases_province(input_file, output_file):
    with open(input_file, "r") as read_obj, open(output_file,'w',newline='') as write_obj:
        csv_reader = reader(read_obj)
        csv_writer = writer(write_obj)
               
        lines=[]
        for line in csv_reader:
            lines.append(line)    

        i=0
        ix=0
        for i in range(0,len(lines[:])-1):
            if lines[i][1]==lines[i+1][1]:
                if ix==0:
                    ix=i
                lines[ix][4:] = np.asfarray(lines[ix][4:],float)+np.asfarray(lines[i+1][4:] ,float)
            else:
                if not ix==0:
                    lines[ix][0]=""
                    csv_writer.writerow(lines[ix])
                    ix=0
                else:
                    csv_writer.writerow(lines[i])
            i+=1    


def download_data(url_dictionary):
    #Lets download the files
    for url_title in url_dictionary.keys():
        urllib.request.urlretrieve(url_dictionary[url_title], "./data/" + url_title)


def load_json(json_file_str):
    # Loads  JSON into a dictionary or quits the program if it cannot.
    try:
        with open(json_file_str, "r") as json_file:
            json_variable = json.load(json_file)
            return json_variable
    except Exception:
        sys.exit("Cannot open JSON file: " + json_file_str)


class Learner(object):
    def __init__(self, country, loss, start_date,dataFiveBefore, predict_range,s_0, a_0, y_0, r_0, d_0):
        self.country = country
        self.loss = loss
        self.start_date = start_date
        self.dataFiveBefore = dataFiveBefore
        self.predict_range = predict_range
        self.s_0 = s_0
        self.a_0 = a_0
        self.y_0 = y_0
        self.r_0 = r_0
        self.d_0 = d_0


    def load_confirmed(self, country):
        df = pd.read_csv('data/time_series_19-covid-Confirmed-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]


    def load_recovered(self, country):
        df = pd.read_csv('data/time_series_19-covid-Recovered-country.csv')
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]


    def load_dead(self, country):
        df = pd.read_csv('data/time_series_19-covid-Deaths-country.csv')
        country_df = df[df['Country/Region'] == country]
        # print(type(country_df))
        # print(type(country_df))
        return country_df.iloc[0].loc[self.start_date:]

    def load_Asymptomatic(self, country):
        df = pd.read_csv('data/Asymptomatic6.csv')
        country_df = df[df['Country/Region'] == country]
        # country_df = df.iloc[0]
        # return country_df.iloc[0].loc[self.start_date:]
        # print(type(pd.DataFrame(df.iloc[0].loc[self.start_date:])))
        # print(df.iloc[0].loc[self.start_date:])
        return country_df.iloc[0].loc[self.start_date:]
        # print(country_df)
        # print('------',country_df.shape,np.shape(pd.DataFrame(country_df).iloc[0:]))
        # print(type(df))
        # print(df.iloc[0].loc[self.start_date:])
        # return country_df.iloc[0].iloc[0:]

    

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

#######
    def predict(self, alpha, beta, gamma, dataFiveBefore, data, illDeath, recovered, death, country, s_0, a_0, y_0, r_0, d_0):
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SAYRD(t, y):
            S = y[0]
            A = y[1]
            Y = y[2]
            R = y[3]
            D = y[4]
            #return [-beta*S, beta*S-0.008638*A-alpha*A, alpha*A-gamma*Y-0.008638*Y-illDeath*Y, gamma*Y-0.008638*R, illDeath*Y]
            return [-beta*S, beta*S-alpha*A, alpha*A-gamma*Y-illDeath*Y, gamma*Y, illDeath*Y]
            # return [100000000*0.012/365-beta*S-0.0082*S/365, beta*S-alpha*A-0.0082*A/365, alpha*A-gamma*Y-illDeath*Y-0.0082*Y/365, gamma*Y-0.0082*R/365, illDeath*Y]
        #extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        ##########
        extended_A = np.concatenate((dataFiveBefore.values, [None] * (size - len(dataFiveBefore.values))))
        extended_Y = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return new_index, extended_A, extended_Y, extended_recovered, extended_death, solve_ivp(SAYRD, [0, size], [s_0,a_0,y_0,r_0, d_0], t_eval=np.arange(0, size, 1))


    def train(self): 
        recovered = self.load_recovered(self.country)
        death = self.load_dead(self.country) 
        data = (self.load_confirmed(self.country) - recovered - death) # Y
        dataFiveBefore = self.load_Asymptomatic(self.country) # A 
        
        # optimal = minimize(loss, [0.2, 0.01, 0.02, 0.02], args=(dataFiveBefore, data, recovered, death, self.s_0, self.a_0, self.y_0, self.r_0, self.d_0), method='L-BFGS-B', 
        # # bounds=[(0.172, 0.222), (0.01, 0.001), (0.02, 1),(0.001, 0.09)]) # alpha beta gamma mu
        # bounds=[(0.172, 0.222), (0.000027,0.04), (0.003,0.4),(0.0005,0.01)]) # alpha beta gamma mu
        optimal = minimize(loss, [0.2, 0.01, 0.02, 0.02], args=(dataFiveBefore, data, recovered, death, self.s_0, self.a_0, self.y_0, self.r_0, self.d_0), method='L-BFGS-B', 
        # bounds=[(0.172, 0.222), (0.01, 0.001), (0.02, 1),(0.001, 0.09)]) # alpha beta gamma mu
        bounds=[(0.172, 0.222), (0.00000001,0.4), (0.0000001,0.4),(0.00000001,0.4)]) # alpha beta gamma mu

        print(optimal)
        alpha, beta, gamma, illDeath = optimal.x
        print('alpha: ', alpha, 'beta: ', beta, 'gamma: ', gamma, 'illDeath: ', illDeath)
        new_index, extended_A, extended_Y, extended_recovered, extended_death, prediction = self.predict(alpha, beta, gamma, dataFiveBefore, data, illDeath, recovered, death, self.country, self.s_0, self.a_0, self.y_0, self.r_0, self.d_0)
        # print('S_94: ', prediction.y[0].item(94), 'A_94: ', prediction.y[1].item(94), 'Y_94: ', prediction.y[2].item(94), 'R_0: ', prediction.y[3].item(94), 'D_94: ', prediction.y[4].item(94))
        #df = pd.DataFrame({'A data': extended_A,'Y data': extended_Y, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Susceptible': prediction.y[0], 'Symptotic': prediction.y[2], 'Recovered': prediction.y[3], 'Asymptotic':prediction.y[1]}, index=new_index)
        # print('type of A: ', extended_A)
        # print('A_94: ', extended_A.item((0)))
        df = pd.DataFrame({'Estimated Asymptomatic data': extended_A,'Symptomatic data': extended_Y, 'Recovered data': extended_recovered, 'Death data': extended_death}, index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        # plt.yscale('log')
        ax.set_title(f"{self.country} Real Data + Estimated Asymptomatic data")
        df.plot(ax=ax)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        fig.savefig(f"{self.country}_real.png")

        pop = self.s_0
        df = pd.DataFrame({'Susceptible': prediction.y[0], 'Symptomatic': prediction.y[2], 'Recovered': prediction.y[3], 'Asymptomatic':prediction.y[1], 'Predicted Death by illness': prediction.y[4]}, index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(f"{self.country} Predicted Model")
        plt.ylabel(f'{self.country} Population Proportion')
        plt.xlabel(f'Date (Alpha:'+str(alpha)+', Beta:'+str(beta)+', Gamma:'+str(gamma)+', Mu:'+str(illDeath)+')')
        #plt.grid(True)
        # plt.xlabel('Date %1.3f' %alpha)
        df.plot(ax=ax)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        fig.savefig(f"{self.country}_Pred.png")

        df = pd.DataFrame({'Symptomatic': prediction.y[2]/pop, 'Recovered': prediction.y[3]/pop, 'Asymptomatic':prediction.y[1]/pop, 'Predicted Death by illness': prediction.y[4]/pop}, index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.ylabel(f'{self.country} Population Proportion')
        plt.xlabel(f'Date (Alpha:'+str(alpha)+', Beta:'+str(beta)+', Gamma:'+str(gamma)+', Mu:'+str(illDeath)+')')
        ax.set_title(f"{self.country} Predicted Model without Susceptible Proportion")
        df.plot(ax=ax)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        fig.savefig(f"{self.country}_Pred_NoS.png")

        df = pd.DataFrame({'Susceptible': prediction.y[0], 'Symptomatic': prediction.y[2], 'Recovered': prediction.y[3], 'Asymptomatic':prediction.y[1], 'Predicted Death by illness': prediction.y[4]}, index=new_index)
        # df = pd.DataFrame({'Symptotic': prediction.y[2]/pop, 'Recovered': prediction.y[3]/pop, 'Asymptotic':prediction.y[1]/pop, 'Predicted Death by illness': prediction.y[4]/pop}, index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        plt.yscale('log')
        ax.set_title(f"{self.country} Predicted Model without Susceptible Proportion After log-scale on y-axis")
        plt.ylabel(f'{self.country} Population Proportion by Log Scale')
        plt.xlabel(f'Date (Alpha:'+str(alpha)+', Beta:'+str(beta)+', Gamma:'+str(gamma)+', Mu:'+str(illDeath)+')')
        df.plot(ax=ax)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.minorticks_on()
        plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        fig.savefig(f"{self.country}_Pred_log.png")

        print(f"country={self.country}, beta={beta:.8f}, alpha={alpha:.8f}, gamma={gamma:.8f}, mu:{illDeath:.8f}")


#######
def loss(point, dataFiveBefore, data, recovered, death, s_0, a_0, y_0, r_0, d_0):
    size = len(data)
    beta, gamma, alpha, illDeath = point
    def SAYRD(t, y):
        S = y[0]
        A = y[1]
        Y = y[2]
        R = y[3]
        D = y[4]
        #return [-beta*S, beta*S-0.008638*A-alpha*A, alpha*A-gamma*Y-0.008638*Y-illDeath*Y, gamma*Y-0.008638*R, illDeath*Y]
        return [-beta*S, beta*S-alpha*A, alpha*A-gamma*Y-illDeath*Y, gamma*Y, illDeath*Y]
        # return [100000000*0.012/365-beta*S-0.0082*S/365, beta*S-alpha*A-0.0082*A/365, alpha*A-gamma*Y-illDeath*Y-0.0082*Y/365, gamma*Y-0.0082*R/365, illDeath*Y]
    solution = solve_ivp(SAYRD, [0, size], [s_0,a_0,y_0,r_0,d_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l0 = np.sqrt(np.mean((solution.y[1] - dataFiveBefore)**2)) # A
    l1 = np.sqrt(np.mean((solution.y[2] - data)**2)) # Y
    l2 = np.sqrt(np.mean((solution.y[3] - recovered)**2)) # R
    l3 = np.sqrt(np.mean((solution.y[4] - death)**2)) # D
    # return 0.1 * l0 + 0.5 * l1 + 0.2 * l2 + 0.3 * l3 # R and D are too high
    return 0.3 * l0 + 0.3 * l1 + 0.3 * l2 + 0.3 * l3


def main():

    countries, download, startdate, dataFiveBefore, predict_range , s_0, a_0, y_0, r_0, d_0 = parse_arguments()

    if download:
        data_d = load_json("./data_url.json")
        download_data(data_d)

    sumCases_province('data/time_series_19-covid-Confirmed.csv', 'data/time_series_19-covid-Confirmed-country.csv')
    sumCases_province('data/time_series_19-covid-Recovered.csv', 'data/time_series_19-covid-Recovered-country.csv')
    sumCases_province('data/time_series_19-covid-Deaths.csv', 'data/time_series_19-covid-Deaths-country.csv')

    for country in countries:
        learner = Learner(country, loss, startdate, dataFiveBefore, predict_range, s_0, a_0, y_0, r_0, d_0)
        #try:
        learner.train()
        #except BaseException:
        #    print('WARNING: Problem processing ' + str(country) +
        #        '. Be sure it exists in the data exactly as you entry it.' +
        #        ' Also check date format if you passed it as parameter.')
           

if __name__ == '__main__':
    main()
