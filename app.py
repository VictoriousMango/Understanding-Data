import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, url_for, render_template, request, redirect, send_file
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from codeAsset.correlation import correlation as corr
from codeAsset.multicolinearity import multicolinearity as Mcol
import seaborn as sns

app = Flask(__name__)

@app.route("/FileSave", methods=["POST"])
def FileSave():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename:
            file.save("./Data/Data.csv")
            print(f"File {file.filename} Uploaded")
    return redirect("/")

@app.route("/RemoveFile")
def RemoveFile():
    os.remove("./Data/Data.csv")
    return redirect("/")

@app.route("/weightGeneration", methods=["GET","POST"])
def weightGeneration():
    if request.method == 'POST':
        try:
            df = pd.read_csv("./Data/Data.csv")
            # print("This is what I am looking for" + request)
            featuresName = [i.strip() for i in request.form.get("features").split(",")]
            targetVar = request.form.get("targetVariable").strip()
            columnNames = [[columnName, 0] for columnName in df.columns if columnName in featuresName]
            X = df[featuresName]
            y = df[targetVar]

            # Model
            model = RandomForestRegressor(max_depth=2, random_state=0)
            # Model Fit
            model.fit(X, y)
            for i in range(len(model.feature_importances_)):
                columnNames[i][1] += model.feature_importances_[i]
            return render_template("./weights.html", columnNames=columnNames, featuresName=", ".join(featuresName), targetVar=targetVar)
        except Exception as e:
            print(f"Exception : {e}")
            return redirect(f"/Except/{e}")
    return redirect(f"/Except/{request.method}")


@app.route("/")
def hello_world():
    try:
        df = pd.read_csv("Data/Data.csv")
        nonNumeric = list()        
        try:
            with open("./static/images/pairPlot.png", 'rb') as f:
                f.read()    
        except:
            sns.pairplot(df) 
            plt.savefig('./static/images/pairPlot.png')
        for i in df.columns:
            try:
                pd.to_numeric(df[i])
                nonNumeric.append(f"No Non Numeric Values in {i} column")
            except ValueError:
                nonNumeric.append(f"Non Numeric Values in {i} column")
            except Exception as e:
                print(f"Exception : {e}")
                return redirect(f"/Except/{e}")
        return render_template(
            "FileUploaded.html", 
            headings=df.columns, 
            table_data_head=df.head().to_dict(orient="records"),
            table_data_tail=df.tail().to_dict(orient="records"),
            table_data_describe=df.describe().to_dict(orient="records"),
            table_data_summary=df[df.isnull().any(axis=1)].to_dict(orient="records"),
            table_data_summary_nonNumeric = nonNumeric,
            describe_columns = ["count" , "mean", "std", "min", "25%", "50%", "75%", "max"]            )
    except FileNotFoundError:
        return render_template("uploadFile.html")

@app.route("/correlations")
def correlations():
    try:
        (table_data, headings) = corr()
        return render_template("/correlation.html", table_data=table_data, headings=headings)
    except Exception as e:
        print(f"Exception : {e}")
        return redirect(f"/Except/{e}")

@app.route("/MultiColinearity", methods=["POST", "GET"])
def multicolinearity():
    try:
        featuresName = ""
        if request.method == 'POST' and request.form.get("features") != "":
            featuresName = [i.strip() for i in request.form.get("features").split(",")]
        vif = Mcol(featuresName)
        return render_template(
            "./multicolinearity.html", 
            vif=vif, 
            featuresName=", ".join(featuresName) if request.form.get("features") != "" else ""
            )
    except Exception as e:
        print(f"Exception : {e}")
        return redirect(f"/Except/{e}")

@app.route("/weights")
def weights():
    try:
        df = pd.read_csv("./Data/Data.csv")
        columnNames = [[columnName, 0] for columnName in df.columns]
        return render_template("./weights.html", columnNames=columnNames)
    except Exception as e:
        print(f"Exception : {e}")
        return redirect(f"/Except/{e}")
    pass

@app.route("/Except/<EXCEPTION>")
def expception(EXCEPTION):
    return render_template("./Exception/Exception.html", Exception=EXCEPTION)

INPUT_DIR = "./tif_files"
OUTPUT_FILE = "./Data/weighted_sum.tif"

if __name__ == "__main__":
    app.run(debug=True, port=5000)