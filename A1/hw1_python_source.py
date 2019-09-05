import csv
from pandas import ExcelFile
from pandas import ExcelWriter
import sys
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import interpolate


class HW1_Solutions():
    # Q1
    num_folds = 0

    def q1_sol(self):
        paper_chickness = pow(10, -3)
        mountain_height = 8848

        def paperfold(thickness, mountain_height):
            if (thickness >= mountain_height):
                return self.num_folds
            self.num_folds += 1
            paperfold(thickness*2, mountain_height)
        paperfold(paper_chickness, mountain_height)
        return self.num_folds

    # Q2

    def q2_sol(self):
        a = 0.1
        # print(pow(2.71828182846,(-a)*6.9314718))
        return (math.log(1/2)/(-a))

    # Q3
    q3_final = []

    def q3_sol(self, amount, rate, year):
        if year <= 5:
            self.q3_final.append(amount)
            # print("year %d: %.5f" % (year, amount))
        if year > 5:
            return
        year += 1
        amount *= (1+rate)
        self.q3_sol(amount, rate, year)
        return [round(x) for x in self.q3_final]

    # Q4
    def q4_sol(self, present_value, rate, num_period_arr):
        num_period_arr = [12 * i for i in num_period_arr]
        return ([round(rate*(present_value)/(1-pow((1+rate), (-num_period)))) for num_period in num_period_arr])

    # Q5
    def q5_sol(self):
        invest = 100000
        customers = 100
        grow_rate = 0.01
        profit = 0
        profitarr = [0]
        repay_days = 0
        while profit <= invest:
            profit += 10*customers
            profitarr.append(profit)
            repay_days += 1
            customers = round(customers*(1+grow_rate))
        print(repay_days)
        daysarr = [i for i in range(repay_days+1)][1:]
        plt.plot(daysarr, profitarr[1:])
        plt.plot(repay_days, profitarr[repay_days], '*g')
        plt.plot(1, profitarr[1], '*g')
        plt.xlim(0, repay_days+10)
        plt.ylim(0, profit+500)
        plt.xlabel("Number of Days")
        plt.ylabel("Cumulated Profits")
        plt.annotate('breakeven\n day:%s \n profit:$%d' % (repay_days, profitarr[-1]),
                     xy=(repay_days, profitarr[repay_days]),
                     textcoords="offset points",
                     #  xytext=(repay_days-5, profitarr[repay_days]-5),
                     xytext=(0, -50),
                     ha='center'
                     )
        plt.annotate('initial investment\n day: 1 \n profit:$%d' % profitarr[1],
                     xy=(1, profitarr[1]),
                     textcoords="offset points",
                     #  xytext=(repay_days-5, profitarr[repay_days]-5),
                     xytext=(50, 10),
                     ha='center'
                     )
        plt.show()
        return repay_days

    # Q6
    def q6_sol(self):

        import datetime
        import calendar
        from matplotlib.dates import (YEARLY, DateFormatter,
                                      rrulewrapper, RRuleLocator, drange)
        df = pd.read_excel('ebola_download.xls', sheet_name='Sheet1')
        df['Date'] = pd.to_datetime(df['Date'], format="%Y/%m/%d").dt.date
        # print("Column headings:")
        # print(df.columns)
        formatter = DateFormatter('%m/%d/%y')
        case_arr = list(df['Cases'])
        death_arr = list(df['Death'])
        date_arr = list(df['Date'])
        # case_interpfunc = interpolate.interp1d(date_arr, case_arr)
        # death_interpfunc = interpolate.interp1d(date_arr,death_arr)
        # print(case_interpfunc)
        # dt = datetime.datetime(2014, 3, 22)
        # end = datetime.datetime(2014, 11, 13)
        # step = datetime.timedelta(days=1)
        # print(date_arr)
        # date_new = []
        # while dt < end:
        #     # print(dt.strftime('%Y-%m-%d'))
        #     date_new.append(dt)
        #     dt += step
        # # print(date_arr)
        # # print(date_new)
        # # test = case_interpfunc(date_new)
        # # print(test)
        case_threshold, death_threshold = {100: None, 500: None, 1000: None, 2000: None, 5000: None}, {
            100: None, 500: None, 1000: None, 2000: None, 5000: None}
        for i in range(len(date_arr)):
            if case_arr[i] >= 100 and case_threshold[100] == None:
                case_threshold[100] = i
            if case_arr[i] >= 500 and case_threshold[500] == None:
                case_threshold[500] = i
            if case_arr[i] >= 1000 and case_threshold[1000] == None:
                case_threshold[1000] = i
            if case_arr[i] >= 2000 and case_threshold[2000] == None:
                case_threshold[2000] = i
            if case_arr[i] >= 5000 and case_threshold[5000] == None:
                case_threshold[5000] = i
            if death_arr[i] >= 100 and death_threshold[100] == None:
                death_threshold[100] = i
            if death_arr[i] >= 500 and death_threshold[500] == None:
                death_threshold[500] = i
            if death_arr[i] >= 1000 and death_threshold[1000] == None:
                death_threshold[1000] = i
            if death_arr[i] >= 2000 and death_threshold[2000] == None:
                death_threshold[2000] = i
            if death_arr[i] >= 5000 and death_threshold[5000] == None:
                death_threshold[5000] = i

        # Plot part
        fig, ax = plt.subplots()
        plt.plot_date(date_arr, case_arr,'-',
                      label='# ebola cases')
        plt.plot_date(date_arr, death_arr,'-',
                      label='# ebola deaths')
        case_limitarr = []
        for i in case_threshold.keys():
            case_limitarr.append(
                (date_arr[case_threshold[i]], case_arr[case_threshold[i]]))
        death_limitarr = []
        for i in death_threshold.keys():
            death_limitarr.append(
                (date_arr[death_threshold[i]], death_arr[death_threshold[i]]))
        plt.plot_date([i[0] for i in case_limitarr], [i[1]
                                                      for i in case_limitarr], marker='o', color='black')
        plt.plot_date([i[0] for i in death_limitarr], [i[1]
                                                       for i in death_limitarr], marker='o', color='red')
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(rotation=30, labelsize=10)
        plt.legend()


        dt = datetime.datetime(2014, 3, 22)
        end = datetime.datetime(2014, 11, 13)
        step = datetime.timedelta(days=1)
        # print(date_arr)
        date_new = []
        while dt < end:
            # print(dt.strftime('%Y-%m-%d'))
            date_new.append(dt)
            dt += step
        # print(np.asarray(date_new).shape)
        # print(np.asarray(case_arr).shape)
        # f = interpolate.interp1d(date_arr,case_arr)
        # plt.plot(date_new,f(date_new))
        plt.show()
        return case_limitarr, death_limitarr

    # Q7
    def q7_sol(self):
        df = pd.read_excel('ebola_download.xls', sheet_name='Sheet1')
        case_arr = list(df['Cases'])
        death_arr = list(df['Death'])
        date_arr = list(df['Date'])
        case_rate = 0
        death_rate = 0
        for i in range(1, len(case_arr)):
            case_rate += (case_arr[i]-case_arr[i-1])/case_arr[i-1]
        for i in range(1, len(death_arr)):
            death_rate += (death_arr[i]-death_arr[i-1])/death_arr[i-1]
        case_rate /= (len(case_arr)-1)
        death_rate /= (len(death_arr)-1)
        print("%.5f%%" % (case_rate*100))
        print("%.5f%%" % (death_rate*100))

    # Q8
    def q8_sol(self):
        df = pd.read_excel('ebola_download.xls', sheet_name='Sheet1')
        case_arr = list(df['Cases'])
        death_arr = list(df['Death'])
        date_arr = list(df['Date'])
        plt.plot(case_arr, death_arr)
        death2case_ratio = 0
        for i in range(len(case_arr)):
            death2case_ratio += death_arr[i]/case_arr[i]
        death2case_ratio /= len(case_arr)
        print("Ratio: %.5f%%" % (death2case_ratio*100))
        plt.xlabel("# Cases")
        plt.ylabel("# Deaths")
        plt.show()
    # Q9
    def q9_sol(self):
        spyfile = open("SPY.csv", "r")
        spy_reader = csv.reader(spyfile)
        spy_arr = [(rows[4]) for rows in spy_reader][1:]
        spyfile.close()
        tltfile = open("TLT.csv", "r")
        tlt_reader = csv.reader(tltfile)
        tlt_arr = [rows[4] for rows in tlt_reader][1:]
        tltfile.close()
        date_arr = [i for i in range(len(spy_arr))]
        spy_scale = 100/float(spy_arr[0])
        tlt_scale = 100/float(tlt_arr[0])
        spy_close = [float(i)*spy_scale for i in spy_arr]
        tlt_close = [float(i)*tlt_scale for i in tlt_arr]
        plt.plot(date_arr, spy_close, label="SPY Curve")
        plt.plot(date_arr, tlt_close, label="TLT Curve")
        plt.ylabel("Closing Prices ($)")
        plt.xlabel("Days since 2014-01-01")
        plt.legend()
        plt.plot(date_arr, tlt_close)
        plt.show()

    def q_10sol(self):
        spy_df = pd.read_csv("SPY.csv")
        # spy_dayarr = list(spy_df['Date'])
        spy_arr = list(spy_df['Close'])
        tlt_df = pd.read_csv("TLT.csv")
        # tlt_dayarr = list(tlt_df['Date'])
        tlt_arr = list(tlt_df['Close'])
        spy_scale = 100/float(spy_arr[0])
        tlt_scale = 100/float(tlt_arr[0])
        spy_close = [float(i)*spy_scale for i in spy_arr]
        tlt_close = [float(i)*tlt_scale for i in tlt_arr]

        def find_returns(arr):
            curr_max = 0
            curr_min = 999
            tot = 0
            for i in range(1, len(arr)):
                tmp_ret = (arr[i]/arr[i-1])-1
                if tmp_ret > curr_max:
                    curr_max = tmp_ret
                if tmp_ret < curr_min:
                    curr_min = tmp_ret
                tot += tmp_ret
            avg = tot/(len(arr))
            return curr_max, curr_min, avg
        spy_max, spy_min, spy_avg = find_returns(spy_close)
        tlt_max, tlt_min, tpt_avg = find_returns(tlt_close)
        print(spy_max*100, spy_min*100, spy_avg*100)
        print(tlt_max*100, tlt_min*100, tpt_avg*100)
        return


if __name__ == "__main__":
    all_solution = HW1_Solutions()
    # q1 = all_solution.q1_sol()
    # print("-----Question 1-----\n", q1)
    # q2 = all_solution.q2_sol()
    # print("-----Question 2-----\n", q2)
    # # deposit = 100 rate = 0.05 year = 0
    # q3 = all_solution.q3_sol(100, 0.05, 0)
    # print("-----Question 3-----\n", q3)
    # q4 = all_solution.q4_sol(20000,0.01,[1,2,3])
    # print("-----Question 4-----\n", q4)
    # q5 = all_solution.q5_sol()
    # print("-----Question 5-----\n", q5)
    # q6_1, q6_2 = all_solution.q6_sol()
    # print("-----Question 6-----")
    # print(q6_1)
    # print(q6_2)
    # print("-----Question 7-----")
    # all_solution.q7_sol()
    # print("-----Question 8-----")
    # all_solution.q8_sol()
    # print("-----Question 9-----")
    # all_solution.q9_sol()
    print("-----Question 10-----")
    all_solution.q_10sol()
