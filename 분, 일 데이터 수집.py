from BTB.kiwoom import *
import time
import datetime


########################################################################################################################

# 로그인
kiwoom = Kiwoom()
kiwoom.CommConnect(block=True)

# 주식계좌 로드
accounts = kiwoom.GetLoginInfo("ACCNO")
stock_account = accounts[0]

# 조건식
kiwoom.GetConditionLoad()
conditions = kiwoom.GetConditionNameList()
condition_index1, TBB = conditions[0]
code1 = kiwoom.SendCondition("0101", TBB, condition_index1, 0)

# 날짜
ye = str(datetime.datetime.today().year)
mo = str(datetime.datetime.today().month).rjust(2, '0')
da = str(datetime.datetime.today().day).rjust(2, '0')

# 3분봉 다운
for i in code1:

    df = kiwoom.block_request("opt10081",
                              종목코드=i,
                              기준일자=ye + mo + da,
                              수정주가구분=1,
                              output="주식일봉차트조회",
                              next=0)

    df.to_excel("DAY"+ i + ye + mo + da + ".xlsx")
    time.sleep(0.2)

    df = kiwoom.block_request("opt10080",
                              종목코드=i,
                              틱범위=3,
                              수정주가구분=1,
                              output="주식3분봉차트조회",
                              next=0)

    df.to_excel("MIN"+i + ye + mo + da + ".xlsx")
    time.sleep(0.2)
