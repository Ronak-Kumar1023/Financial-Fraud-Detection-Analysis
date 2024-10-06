import pandas as pd
import matplotlib.pyplot as plt

def exercise_0(file):
    return pd.read_csv(file)

def exercise_1(df):
    return list(df)

def exercise_2(df, k):
    return df.head(k)

def exercise_3(df, k):
    return df.sample(n=k)

def exercise_4(df):
    return df['type'].unique()

def exercise_5(df):
    return df['nameDest'].value_counts().head(10)

def exercise_6(df):
    return df[df['isFraud'] == 1]

def exercise_7(df):
    df1 = df.groupby('nameOrig')['nameDest'].agg(['nunique'])
    df1.sort_values(by=('nunique'), ascending=False, inplace=True)
    return df1

def visual_1(df):
    def transaction_counts(df):
        return df['type'].value_counts()
    
    def transaction_counts_split_by_fraud(df):
        return df.groupby(by=['type', 'isFraud']).size()

    fig, axs = plt.subplots(2, figsize=(6,10))
    transaction_counts(df).plot(ax=axs[0], kind='bar', color='blue')
    axs[0].set_title('Transactions by Type')
    axs[0].set_xlabel('Transaction Types')
    axs[0].set_ylabel('Count')
    transaction_counts_split_by_fraud(df).plot(ax=axs[1], kind='bar', color='blue')
    axs[1].set_title('Transaction Types by Fraud Status')
    axs[1].set_xlabel('Transaction Type, Split by Fraud')
    axs[1].set_ylabel('Count')
    fig.suptitle('Transaction Types')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    for ax in axs:
      for p in ax.patches:
          ax.annotate(p.get_height(), (p.get_x(), p.get_height()))
    plt.show()
    return 'This bar chart provides actionable insights by exploring the existence of fraud among the transaction types. From this, we can conclude that the only fraudulent transactions are CASH_OUT and TRANSFER. The company can take action by investigating these specific transaction types and adding layers of security.'

df = exercise_0("transactions.csv")

visual_1(df)

def visual_2(df):
    def query(df):
        df['Origin Delta'] = df['oldbalanceOrg'] -	df['newbalanceOrig']
        df['Destination Delta'] = df['oldbalanceDest'] -	df['newbalanceDest']
        return df[df['type']=='CASH_OUT']
    plot = query(df).plot.scatter(x='Origin Delta',y='Destination Delta', alpha=.5, color='blue')
    plot.set_title('Origin vs. Destination Balance Delta for Cash Out Transactions')
    plot.set_xlim(left=-1e3, right=1e3)
    plot.set_ylim(bottom=-1e3, top=1e3)
    return 'This chart showcases patterns between origin and destination account balances during cash out transactions. Instantaneous settlement is shown by the y=-x line, which happens when an increase in the destination balance exactly matches a decrease in the origin balance. It is possible to identify outliers that point to fraud by looking at trends in the data. These include transactions in which the balance changes at the origin and destination differ significantly.'

visual_2(df)

def exercise_custom(df):
    return df.groupby(['isFlaggedFraud', 'isFraud']).size()

def visual_custom(df):
    labels = ['True Negative (0, 0)', 'False Negative (0, 1)', 'False Positive (1, 0)', 'True Positive (1, 1)']
    exercise_custom(df).plot.pie(figsize=(7, 7), labels=labels, autopct='%1.1f%%')

    plt.title('Fraud Detection Scenarios Distribution')
    plt.ylabel('')
    plt.tight_layout()

    return 'This pie chart shows the proportion of different fraud detection scenarios in order to analyze the efficiency of the fraud detection system. Although the system is able to flag 99.9% of fraud that is actualy fraudlent activity (True Negative), it is unable to flag 0.1% of actual fraudulent activity (False Negative).'
    
visual_custom(df)


