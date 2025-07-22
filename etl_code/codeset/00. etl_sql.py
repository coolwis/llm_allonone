#!/usr/bin/env python
# coding: utf-8

# # 1. 라이브러리 선언 및 데이터 불러오기

# In[1]:


# 기본 데이터 처리 및 시각화 라이브러리
import pandas as pd
from datetime import datetime, timedelta


# 날짜 계산 라이브러리
import sys

# postgresql 접속 라이브러리
import psycopg2 
from sqlalchemy import create_engine 


def next_day(input_date, delta):
    input_date = str(input_date)
    delta = int(delta)
    # 입력된 날짜를 YYYYMMDD 형식에서 파싱
    date_object = datetime.strptime(input_date, '%Y%m%d')
    
    # 하루를 더한 후에 문자열로 변환하여 반환
    next_date = (date_object + timedelta(days=delta)).strftime('%Y%m%d')
    return next_date


paramData = pd.read_csv("../dataset/datasource.csv")


# incremental/full 옵션 처리
targetSeg = "famili"

# datasource.csv에서 mode 값 읽기
mode = paramData.loc[(paramData.seg == targetSeg) & (paramData.param == "mode")].data.values[0]


conPrefix = paramData.loc[ (paramData.seg==targetSeg) &\
                (paramData.param=="dbprefix")].data.values[0]

conUserId = paramData.loc[ (paramData.seg==targetSeg) &\
                (paramData.param=="dbuser")].data.values[0]

conUserPw = paramData.loc[ (paramData.seg==targetSeg) &\
                (paramData.param=="dbpw")].data.values[0]

coDbIp = paramData.loc[ (paramData.seg==targetSeg) &\
                (paramData.param=="dbip")].data.values[0]

coDbPort = paramData.loc[ (paramData.seg==targetSeg) &\
                (paramData.param=="dbport")].data.values[0]

coDbName = paramData.loc[ (paramData.seg==targetSeg) &\
                (paramData.param=="dbname")].data.values[0]



dbConn = "{}://{}:{}@{}:{}/{}".format(conPrefix,conUserId,conUserPw,coDbIp,coDbPort,coDbName)

# In[13]:
# 현재 날짜를 가져옴
currDate = datetime.now()
# 2년 전의 날짜를 계산
strtDate = currDate - timedelta(days=365*2)
# 결과 출력 (YYYY-MM-DD 형식)
strtDate = strtDate.strftime('%Y-%m-%d')
strtDate

# 현재 날짜를 YYYY-MM-DD 형식의 문자열로 가져오기
currentDate = datetime.now().strftime('%Y-%m-%d')
currentDate

engine = create_engine(dbConn) 


# PostgreSQL 연결 설정
conn = psycopg2.connect(
    dbname=coDbName,
    user=conUserId,
    password=conUserPw,
    host=coDbIp,
    port=coDbPort,
)


# In[15]:


# 커서 생성
cur = conn.cursor()

# SQL 쿼리 실행
sql = """
SELECT A.ID
     , A.MEMB_ID
     , CASE
           WHEN CHAR_LENGTH(TD.DEVICE_MODEL) != 0 THEN (TD.DEVICE_DESC || '(' || TD.DEVICE_MODEL || ')')
           ELSE TD.DEVICE_DESC END AS DEVICE_DESC
     , FFMD_USER_NAME(A.MEMB_ID) AS FULLNAME
     , FFMD_LOCAL_DATETIME(A.GET_TIME::TIMESTAMP, 'GMT+9', 'yyyy-mm-dd', 'hh:mm:ss') AS GET_TIME
     , A.GLUCOSEDATA
     , TC.CODE_NAME AS WHEN_EAT
     , FFMD_LOCAL_DATETIME(A.CREATE_DTTM, 'GMT+9', 'yyyy-mm-dd', 'hh:mm:ss') AS CREATE_DTTM
FROM (
    SELECT A.ID
           , A.MEMB_ID
           , A.GET_TIME
           , A.BLS_VALUE AS GLUCOSEDATA
           , A.TAG1 AS WHEN_EAT
           , A.DEVICE_ID
           , A.CREATE_DTTM
      FROM TCMS_DAT_BLS A
      WHERE 1 = 1
        AND TO_DATE(FFMD_LOCAL_DATETIME(GET_TIME::TIMESTAMP, 'GMT+9', 'yyyy-mm-dd', 'hh:mm:ss'),
                    'YYYY-MM-DD') BETWEEN TO_DATE(%s, 'yyyy-mm-dd') AND TO_DATE(%s, 'yyyy-mm-dd')
      ORDER BY GET_TIME DESC
) A
LEFT JOIN TCMS_DEVICE TD ON A.DEVICE_ID = TD.DEVICE_ID
INNER JOIN TFMD_CODE TC ON TC.GROUP_ID = (
    SELECT GROUP_ID
    FROM TFMD_GROUP_CODE
    WHERE GROUP_VALUE = 'CMS_BLS_TAG_1'
) AND A.WHEN_EAT = TC.CODE_VALUE
ORDER BY GET_TIME DESC
"""

# 쿼리 실행
cur.execute(sql, (strtDate, currentDate))

# 결과 가져오기
rows = cur.fetchall()

# 결과를 DataFrame으로 변환
dataset = pd.DataFrame(rows, columns=[desc[0] for desc in cur.description])


# In[21]:


dbConnPg = "{}://{}:{}@{}:{}/{}".format(conPrefix,conUserId,conUserPw,coDbIp,coDbPort,coDbName)


# In[22]:


table_name = paramData.loc[(paramData.seg==targetSeg) & (paramData.param=="glucose")].data.values[0]

if mode == 'full':
    # 전체 데이터 저장
    dataset.to_sql(table_name, dbConnPg, if_exists='replace', index=False)
else:
    # 증분 데이터 저장: 기존 테이블에서 가장 최근 날짜 이후 데이터만 추가
    with engine.connect() as conn:
        result = conn.execute(f"SELECT MAX(\"GET_TIME\") FROM {table_name}")
        last_time = result.scalar()
    if last_time:
        # GET_TIME이 문자열일 경우 datetime으로 변환
        try:
            last_time = pd.to_datetime(last_time)
        except:
            pass
        incremental_data = dataset[pd.to_datetime(dataset['GET_TIME']) > last_time]
        if not incremental_data.empty:
            incremental_data.to_sql(table_name, dbConnPg, if_exists='append', index=False)
    else:
        dataset.to_sql(table_name, dbConnPg, if_exists='replace', index=False)

from datetime import datetime

now = datetime.now()

print("runing etl time and date:", now)