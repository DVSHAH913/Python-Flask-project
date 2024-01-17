from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import ta

app = Flask(__name__)

# MySQL Connection
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'devshree090103'
app.config['MYSQL_DB'] = 'stock_price'
mysql = MySQL(app)

@app.route('/home')
def home():
    return render_template('home.html')
@app.route('/news')
def news():
    return render_template('news.html')
@app.route('/learn')
def learn():
    return render_template('learn.html')
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/prediction', methods=['GET','POST'])
def prediction():  
    if (request.method=='GET'):
        cursor=mysql.connection.cursor()
        cursor.execute("SELECT DISTINCT script FROM dailyprice")
        scripts = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return render_template('prediction.html',scripts=scripts)
    elif (request.method=='POST'):
        script_name = request.form['script_name'] 
        # Get Data from MySQL
        cur = mysql.connection.cursor()
        cur.execute(f"SELECT Date, Script, Open, High, Low, Close, Volume FROM dailyprice WHERE script='{script_name}' ORDER BY date")
        rows = cur.fetchall()
        cur.close()
        
        df = pd.DataFrame(rows, columns=['Date', 'Script','Open','High','Low','Close','Volume'])
        df['7 EH']=df['High'].ewm(span=7).mean()
        df['7 LH']=df['Low'].ewm(span=7).mean()
        df['High_divisional']=(df['High'].ewm(span=7).mean()+df['High'].ewm(span=14).mean()+df['High'].ewm(span=21).mean())/3
        df['Low_divisional']=(df['Low'].ewm(span=7).mean()+df['Low'].ewm(span=14).mean()+df['Low'].ewm(span=21).mean())/3
        sma = df['Close'].rolling(window=7).mean()
        std = df['Close'].rolling(window=7).std()
        df['Bollinger Upper'] = sma + 2*std
        df['Bollinger Middle'] = sma
        df['Bollinger Lower'] = sma - 2*std
        rsi = ta.momentum.RSIIndicator(df['Close'], window=7)
        df['RSI'] = rsi.rsi()
        df['Stock_occ'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=7)
        df['Stock_sig'] = df['Stock_occ'].rolling(window=3).mean()
        df['MACD'] = ta.trend.macd(df['Close'], window_slow=3, window_fast=1)
        df['Signal line'] = ta.trend.macd_signal(df['Close'], window_slow=3, window_fast=3, window_sign=3)
        df1=df.tail(1)
        
        df = df.fillna(df.mean())
        X = df[['High_divisional', 'Bollinger Middle', 'RSI', 'Stock_occ','Stock_sig', 'MACD','Signal line']]
        y = df['Close']
        # split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        # train the regression model
        reg_model = LinearRegression()
        reg_model.fit(X_train, y_train)
        # train the classification model
        clf_model = RandomForestClassifier(n_estimators=100)
        clf_model.fit(X_train, y_train.apply(lambda x: 1 if x > y_train.mean() else 0))
        # make a prediction for tomorrow's price using the regression model
        tomorrow_features = X.tail(1)
        tomorrow_price = reg_model.predict(tomorrow_features)[0]
        # make a prediction for whether to buy or sell using the classification model
        buy_or_sell = clf_model.predict(tomorrow_features)[0]
        if buy_or_sell == 1:
            Signal="Buy"
        else:
            Signal="Sell"
        #evaluate the accuracy of the classification model
        y_pred = clf_model.predict(X_test)
        accuracy = accuracy_score(y_test.apply(lambda x: 1 if x > y_train.mean() else 0), y_pred)

    return render_template('prediction.html', script_name=script_name, df1=df1.to_html(classes="table table-bordered"),tomorrow_price=tomorrow_price, Signal=Signal,accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
